"""
Lambda Producer: Streams synthetic transactions to Kinesis
Simulates real-time transaction flow for testing
"""

import json
import boto3
import os
import logging
from datetime import datetime
from typing import Dict, List, Any
import base64

logger = logging.getLogger()
logger.setLevel(logging.INFO)

kinesis_client = boto3.client('kinesis')
s3_client = boto3.client('s3')

KINESIS_STREAM_NAME = os.environ.get('KINESIS_STREAM_NAME')
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '100'))
MAX_RECORDS_PER_INVOCATION = int(os.environ.get('MAX_RECORDS', '500'))


def lambda_handler(event, context):
    """
    Lambda handler for streaming transactions
    Can be triggered by:
    - EventBridge (scheduled)
    - Manual invocation with test data
    - S3 event (when new data uploaded)
    """
    
    try:
        logger.info(f"Starting producer lambda")
        logger.info(f"Event: {json.dumps(event)}")
        
        # Check if stream name is configured
        if not KINESIS_STREAM_NAME:
            raise ValueError("KINESIS_STREAM_NAME environment variable not set")
        
        # Determine source of transactions
        if 'Records' in event and len(event['Records']) > 0:
            # Triggered by S3 event - get transactions from uploaded file
            transactions = handle_s3_event(event)
        elif 'transactions' in event:
            # Manual invocation with test data
            transactions = event['transactions']
        else:
            # Default: generate sample transactions for testing
            transactions = generate_sample_transactions()
        
        # Stream transactions to Kinesis
        total_sent = stream_transactions(transactions)
        
        response = {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Successfully streamed transactions',
                'transactions_sent': total_sent,
                'stream_name': KINESIS_STREAM_NAME
            })
        }
        
        logger.info(f"Producer completed: {total_sent} transactions sent")
        return response
        
    except Exception as e:
        logger.error(f"Error in producer lambda: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }


def handle_s3_event(event: Dict) -> List[Dict]:
    """Handle S3 event trigger - read parquet file"""
    try:
        import pyarrow.parquet as pq
        import io
        
        record = event['Records'][0]
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        
        logger.info(f"Reading from S3: s3://{bucket}/{key}")
        
        # Get object from S3
        response = s3_client.get_object(Bucket=bucket, Key=key)
        parquet_data = response['Body'].read()
        
        # Read parquet
        table = pq.read_table(io.BytesIO(parquet_data))
        df = table.to_pandas()
        
        # Convert to list of dicts
        transactions = df.head(MAX_RECORDS_PER_INVOCATION).to_dict('records')
        
        # Convert timestamps to strings
        for txn in transactions:
            if 'timestamp' in txn:
                txn['timestamp'] = str(txn['timestamp'])
        
        logger.info(f"Read {len(transactions)} transactions from S3")
        return transactions
        
    except Exception as e:
        logger.error(f"Error reading from S3: {str(e)}")
        raise


def generate_sample_transactions() -> List[Dict]:
    """Generate sample transactions for testing"""
    sample_transactions = []
    
    for i in range(10):
        transaction = {
            'transaction_id': f'TEST_{i}_{int(datetime.now().timestamp())}',
            'timestamp': datetime.now().isoformat(),
            'instrument_id': 'GB000TEST0001',
            'trader_id': 'LEI_TEST_TRADER',
            'counterparty_id': 'LEI_TEST_COUNTERPARTY',
            'price': 100.0 + i,
            'quantity': 1000,
            'side': 'BUY' if i % 2 == 0 else 'SELL',
            'order_type': 'LIMIT',
            'trading_venue': 'XLON',
            'is_algo_trade': False,
            'is_hft': False,
            'order_status': 'EXECUTED',
            'is_manipulation': False,
            'manipulation_type': 'NONE'
        }
        sample_transactions.append(transaction)
    
    logger.info(f"Generated {len(sample_transactions)} sample transactions")
    return sample_transactions


def stream_transactions(transactions: List[Dict]) -> int:
    """Stream transactions to Kinesis in batches"""
    
    total_sent = 0
    batch = []
    
    for txn in transactions:
        # Prepare Kinesis record
        record = {
            'Data': json.dumps(txn, default=str),
            'PartitionKey': txn.get('instrument_id', 'default')
        }
        batch.append(record)
        
        # Send batch when full
        if len(batch) >= BATCH_SIZE:
            send_batch(batch)
            total_sent += len(batch)
            batch = []
    
    # Send remaining records
    if batch:
        send_batch(batch)
        total_sent += len(batch)
    
    return total_sent


def send_batch(records: List[Dict]) -> None:
    """Send a batch of records to Kinesis"""
    try:
        response = kinesis_client.put_records(
            StreamName=KINESIS_STREAM_NAME,
            Records=records
        )
        
        failed_count = response.get('FailedRecordCount', 0)
        
        if failed_count > 0:
            logger.warning(f"Failed to send {failed_count} records")
            # Log failed records for debugging
            for i, record in enumerate(response.get('Records', [])):
                if 'ErrorCode' in record:
                    logger.error(f"Record {i} failed: {record['ErrorCode']} - {record.get('ErrorMessage')}")
        else:
            logger.info(f"Successfully sent batch of {len(records)} records")
            
    except Exception as e:
        logger.error(f"Error sending batch to Kinesis: {str(e)}")
        raise