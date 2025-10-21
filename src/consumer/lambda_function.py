"""
Lambda Consumer: Reads transactions from Kinesis and validates data flow
Writes processed records to S3 for verification
"""

import json
import boto3
import os
import logging
from datetime import datetime
from typing import List, Dict, Any
import base64

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')

DATA_BUCKET = os.environ.get('DATA_BUCKET')
OUTPUT_PREFIX = os.environ.get('OUTPUT_PREFIX', 'processed')


def lambda_handler(event, context):
    """
    Lambda handler for consuming Kinesis records
    Triggered automatically by Kinesis stream
    """
    
    try:
        logger.info(f"Processing {len(event['Records'])} Kinesis records")
        
        if not DATA_BUCKET:
            raise ValueError("DATA_BUCKET environment variable not set")
        
        # Process Kinesis records
        transactions = process_kinesis_records(event['Records'])
        
        # Validate transactions
        valid_transactions = validate_transactions(transactions)
        
        # Write to S3 for verification
        if valid_transactions:
            output_key = write_to_s3(valid_transactions)
            logger.info(f"Wrote {len(valid_transactions)} transactions to s3://{DATA_BUCKET}/{output_key}")
        
        # Log statistics
        log_statistics(transactions, valid_transactions)
        
        return {
            'statusCode': 200,
            'processed': len(transactions),
            'valid': len(valid_transactions),
            'invalid': len(transactions) - len(valid_transactions)
        }
        
    except Exception as e:
        logger.error(f"Error in consumer lambda: {str(e)}", exc_info=True)
        raise


def process_kinesis_records(kinesis_records: List[Dict]) -> List[Dict]:
    """Extract and decode transactions from Kinesis records"""
    
    transactions = []
    
    for record in kinesis_records:
        try:
            # Decode Kinesis data
            payload = base64.b64decode(record['kinesis']['data']).decode('utf-8')
            transaction = json.loads(payload)
            
            # Add Kinesis metadata
            transaction['kinesis_sequence_number'] = record['kinesis']['sequenceNumber']
            transaction['kinesis_partition_key'] = record['kinesis']['partitionKey']
            transaction['kinesis_arrival_timestamp'] = record['kinesis']['approximateArrivalTimestamp']
            
            transactions.append(transaction)
            
        except Exception as e:
            logger.error(f"Error processing record: {str(e)}")
            continue
    
    logger.info(f"Extracted {len(transactions)} transactions from Kinesis records")
    return transactions


def validate_transactions(transactions: List[Dict]) -> List[Dict]:
    """Validate transaction structure and required fields"""
    
    required_fields = [
        'transaction_id',
        'timestamp',
        'instrument_id',
        'price',
        'quantity',
        'side'
    ]
    
    valid_transactions = []
    
    for txn in transactions:
        is_valid = True
        missing_fields = []
        
        # Check required fields
        for field in required_fields:
            if field not in txn or txn[field] is None:
                is_valid = False
                missing_fields.append(field)
        
        # Validate data types
        if is_valid:
            try:
                assert isinstance(txn['price'], (int, float)) and txn['price'] > 0
                assert isinstance(txn['quantity'], (int, float)) and txn['quantity'] > 0
                assert txn['side'] in ['BUY', 'SELL']
            except (AssertionError, KeyError) as e:
                is_valid = False
                logger.warning(f"Validation failed for transaction {txn.get('transaction_id')}: {str(e)}")
        
        if is_valid:
            valid_transactions.append(txn)
        else:
            logger.warning(f"Invalid transaction {txn.get('transaction_id', 'UNKNOWN')}: missing {missing_fields}")
    
    logger.info(f"Validated: {len(valid_transactions)}/{len(transactions)} transactions valid")
    return valid_transactions


def write_to_s3(transactions: List[Dict]) -> str:
    """Write validated transactions to S3"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_id = datetime.now().strftime('%Y%m%d_%H')
    
    output_key = f"{OUTPUT_PREFIX}/batch_{batch_id}/transactions_{timestamp}.json"
    
    # Prepare data
    output_data = {
        'batch_timestamp': datetime.now().isoformat(),
        'record_count': len(transactions),
        'transactions': transactions
    }
    
    # Write to S3
    s3_client.put_object(
        Bucket=DATA_BUCKET,
        Key=output_key,
        Body=json.dumps(output_data, default=str, indent=2),
        ContentType='application/json'
    )
    
    return output_key


def log_statistics(all_transactions: List[Dict], valid_transactions: List[Dict]):
    """Log processing statistics"""
    
    stats = {
        'total_received': len(all_transactions),
        'valid': len(valid_transactions),
        'invalid': len(all_transactions) - len(valid_transactions),
        'validation_rate': len(valid_transactions) / len(all_transactions) if all_transactions else 0
    }
    
    # Count by side
    buy_count = sum(1 for t in valid_transactions if t.get('side') == 'BUY')
    sell_count = sum(1 for t in valid_transactions if t.get('side') == 'SELL')
    
    stats['buy_count'] = buy_count
    stats['sell_count'] = sell_count
    
    # Count manipulation flags
    manipulation_count = sum(1 for t in valid_transactions if t.get('is_manipulation', False))
    stats['manipulation_flagged'] = manipulation_count
    
    logger.info(f"Processing statistics: {json.dumps(stats)}")

