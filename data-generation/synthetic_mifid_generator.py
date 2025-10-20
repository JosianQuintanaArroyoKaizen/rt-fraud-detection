"""
Synthetic MiFID II Transaction Generator
Generates labeled training data for market abuse detection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import random
from pathlib import Path

fake = Faker()
random.seed(42)
np.random.seed(42)


class SyntheticMiFIDGenerator:
    """Generate realistic MiFID II transaction data with manipulation patterns"""
    
    def __init__(self, num_transactions=100000, manipulation_rate=0.10):
        self.num_transactions = num_transactions
        self.manipulation_rate = manipulation_rate
        self.num_manipulations = int(num_transactions * manipulation_rate)
        self.num_normal = num_transactions - self.num_manipulations
        
        # Market parameters
        self.instruments = self._generate_instruments()
        self.traders = self._generate_traders()
        self.venues = ['XLON', 'XPAR', 'XFRA', 'XAMS', 'XMIL']
        
        # Trading hours: 08:00 to 16:30 London time
        self.market_open = 8
        self.market_close = 16.5
        
    def _generate_instruments(self, num=50):
        """Generate synthetic ISINs"""
        instruments = []
        for i in range(num):
            isin = f"GB{fake.bothify(text='??########')}{i:02d}"
            instruments.append({
                'isin': isin,
                'base_price': np.random.uniform(10, 500),
                'volatility': np.random.uniform(0.01, 0.05)
            })
        return instruments
    
    def _generate_traders(self, num=500):
        """Generate synthetic trader LEIs"""
        return [f"LEI{fake.bothify(text='??##################')}" for _ in range(num)]
    
    def _generate_timestamp(self):
        """Generate random timestamp during trading hours"""
        base_date = datetime.now() - timedelta(days=random.randint(1, 90))
        hour = random.uniform(self.market_open, self.market_close)
        hours = int(hour)
        minutes = int((hour - hours) * 60)
        seconds = random.randint(0, 59)
        microseconds = random.randint(0, 999999)
        
        return base_date.replace(
            hour=hours, 
            minute=minutes, 
            second=seconds, 
            microsecond=microseconds
        )
    
    def _generate_normal_transaction(self):
        """Generate a normal, legitimate transaction"""
        instrument = random.choice(self.instruments)
        trader = random.choice(self.traders)
        counterparty = random.choice([t for t in self.traders if t != trader])
        
        # Normal price variation
        price = instrument['base_price'] * np.random.normal(1.0, instrument['volatility'])
        quantity = random.choice([100, 200, 500, 1000, 2000, 5000])
        
        return {
            'timestamp': self._generate_timestamp(),
            'instrument_id': instrument['isin'],
            'trader_id': trader,
            'counterparty_id': counterparty,
            'price': round(price, 2),
            'quantity': quantity,
            'side': random.choice(['BUY', 'SELL']),
            'order_type': random.choice(['MARKET', 'LIMIT', 'LIMIT', 'LIMIT']),
            'trading_venue': random.choice(self.venues),
            'is_algo_trade': random.choice([True, False, False, False]),
            'is_hft': random.choice([True, False, False, False, False]),
            'order_status': 'EXECUTED',
            'is_manipulation': False,
            'manipulation_type': 'NONE'
        }
    
    def _generate_spoofing_pattern(self):
        """Generate spoofing manipulation pattern"""
        instrument = random.choice(self.instruments)
        trader = random.choice(self.traders)
        base_time = self._generate_timestamp()
        
        transactions = []
        
        # Large fake orders on one side
        for i in range(random.randint(5, 15)):
            fake_order_time = base_time + timedelta(seconds=i*2)
            transactions.append({
                'timestamp': fake_order_time,
                'instrument_id': instrument['isin'],
                'trader_id': trader,
                'counterparty_id': random.choice(self.traders),
                'price': round(instrument['base_price'] * 1.01, 2),
                'quantity': random.randint(5000, 20000),
                'side': 'BUY',
                'order_type': 'LIMIT',
                'trading_venue': random.choice(self.venues),
                'is_algo_trade': True,
                'is_hft': True,
                'order_status': 'CANCELLED',
                'is_manipulation': True,
                'manipulation_type': 'SPOOFING'
            })
        
        # Real execution on opposite side
        real_trade_time = base_time + timedelta(seconds=len(transactions)*2 + 1)
        transactions.append({
            'timestamp': real_trade_time,
            'instrument_id': instrument['isin'],
            'trader_id': trader,
            'counterparty_id': random.choice(self.traders),
            'price': round(instrument['base_price'] * 0.99, 2),
            'quantity': random.randint(1000, 3000),
            'side': 'SELL',
            'order_type': 'MARKET',
            'trading_venue': random.choice(self.venues),
            'is_algo_trade': True,
            'is_hft': True,
            'order_status': 'EXECUTED',
            'is_manipulation': True,
            'manipulation_type': 'SPOOFING'
        })
        
        return transactions
    
    def _generate_wash_trading_pattern(self):
        """Generate wash trading manipulation pattern"""
        instrument = random.choice(self.instruments)
        trader = random.choice(self.traders)
        controlled_account = random.choice([t for t in self.traders if t != trader])
        base_time = self._generate_timestamp()
        
        transactions = []
        num_wash_trades = random.randint(3, 8)
        
        for i in range(num_wash_trades):
            trade_time = base_time + timedelta(seconds=i*5)
            price = round(instrument['base_price'] * np.random.normal(1.0, 0.002), 2)
            quantity = random.choice([1000, 2000, 5000])
            
            # Buy from controlled account
            transactions.append({
                'timestamp': trade_time,
                'instrument_id': instrument['isin'],
                'trader_id': trader,
                'counterparty_id': controlled_account,
                'price': price,
                'quantity': quantity,
                'side': 'BUY',
                'order_type': 'LIMIT',
                'trading_venue': random.choice(self.venues),
                'is_algo_trade': False,
                'is_hft': False,
                'order_status': 'EXECUTED',
                'is_manipulation': True,
                'manipulation_type': 'WASH_TRADING'
            })
            
            # Sell back to controlled account
            transactions.append({
                'timestamp': trade_time + timedelta(microseconds=random.randint(100, 1000)),
                'instrument_id': instrument['isin'],
                'trader_id': controlled_account,
                'counterparty_id': trader,
                'price': price,
                'quantity': quantity,
                'side': 'SELL',
                'order_type': 'LIMIT',
                'trading_venue': random.choice(self.venues),
                'is_algo_trade': False,
                'is_hft': False,
                'order_status': 'EXECUTED',
                'is_manipulation': True,
                'manipulation_type': 'WASH_TRADING'
            })
        
        return transactions
    
    def _generate_layering_pattern(self):
        """Generate layering manipulation pattern"""
        instrument = random.choice(self.instruments)
        trader = random.choice(self.traders)
        base_time = self._generate_timestamp()
        
        transactions = []
        num_layers = random.randint(5, 10)
        
        # Multiple orders at different price levels
        for i in range(num_layers):
            layer_time = base_time + timedelta(seconds=i)
            price = round(instrument['base_price'] * (1.0 + (i * 0.001)), 2)
            
            transactions.append({
                'timestamp': layer_time,
                'instrument_id': instrument['isin'],
                'trader_id': trader,
                'counterparty_id': random.choice(self.traders),
                'price': price,
                'quantity': random.randint(1000, 5000),
                'side': 'BUY',
                'order_type': 'LIMIT',
                'trading_venue': random.choice(self.venues),
                'is_algo_trade': True,
                'is_hft': True,
                'order_status': 'CANCELLED',
                'is_manipulation': True,
                'manipulation_type': 'LAYERING'
            })
        
        # One real trade on opposite side
        real_trade_time = base_time + timedelta(seconds=num_layers + 1)
        transactions.append({
            'timestamp': real_trade_time,
            'instrument_id': instrument['isin'],
            'trader_id': trader,
            'counterparty_id': random.choice(self.traders),
            'price': round(instrument['base_price'] * 0.995, 2),
            'quantity': random.randint(2000, 8000),
            'side': 'SELL',
            'order_type': 'MARKET',
            'trading_venue': random.choice(self.venues),
            'is_algo_trade': True,
            'is_hft': True,
            'order_status': 'EXECUTED',
            'is_manipulation': True,
            'manipulation_type': 'LAYERING'
        })
        
        return transactions
    
    def generate(self):
        """Generate complete dataset with normal and manipulation transactions"""
        print(f"Generating {self.num_transactions} synthetic MiFID II transactions...")
        print(f"Normal: {self.num_normal} | Manipulation: {self.num_manipulations}")
        
        all_transactions = []
        
        # Generate normal transactions
        print("Generating normal transactions...")
        for i in range(self.num_normal):
            all_transactions.append(self._generate_normal_transaction())
            if (i + 1) % 10000 == 0:
                print(f"  Generated {i + 1}/{self.num_normal} normal transactions")
        
        # Generate manipulation patterns
        print("Generating manipulation patterns...")
        manipulation_types = ['spoofing', 'wash_trading', 'layering']
        manipulations_per_type = self.num_manipulations // 3
        
        # Spoofing (5% of total)
        spoofing_patterns = manipulations_per_type // 10
        for i in range(spoofing_patterns):
            all_transactions.extend(self._generate_spoofing_pattern())
        
        # Wash Trading (3% of total)
        wash_trading_patterns = manipulations_per_type // 6
        for i in range(wash_trading_patterns):
            all_transactions.extend(self._generate_wash_trading_pattern())
        
        # Layering (2% of total)
        layering_patterns = manipulations_per_type // 11
        for i in range(layering_patterns):
            all_transactions.extend(self._generate_layering_pattern())
        
        print(f"Total transactions generated: {len(all_transactions)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_transactions)
        
        # Add derived fields
        df['transaction_id'] = [f"TXN{fake.bothify(text='##########')}" for _ in range(len(df))]
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_business_hours'] = ((df['hour_of_day'] >= 8) & (df['hour_of_day'] <= 16)).astype(int)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate order_id (link cancelled orders)
        df['order_id'] = [f"ORD{fake.bothify(text='##########')}" for _ in range(len(df))]
        
        # Add bid/ask spreads
        df['bid_price'] = df['price'] * np.random.uniform(0.998, 0.999, len(df))
        df['ask_price'] = df['price'] * np.random.uniform(1.001, 1.002, len(df))
        df['bid_size'] = np.random.choice([100, 500, 1000, 2000], len(df))
        df['ask_size'] = np.random.choice([100, 500, 1000, 2000], len(df))
        
        # Round prices
        df['bid_price'] = df['bid_price'].round(2)
        df['ask_price'] = df['ask_price'].round(2)
        
        print("\nDataset Summary:")
        print(f"Total Transactions: {len(df)}")
        print(f"Normal: {len(df[df['is_manipulation'] == False])}")
        print(f"Manipulation: {len(df[df['is_manipulation'] == True])}")
        print(f"\nManipulation Breakdown:")
        print(df[df['is_manipulation'] == True]['manipulation_type'].value_counts())
        print(f"\nDate Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def save(self, df, output_path):
        """Save dataset to parquet"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"\nSaved to: {output_path}")
        print(f"File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")


def main():
    """Generate and save synthetic MiFID II dataset"""
    
    # Configuration
    NUM_TRANSACTIONS = 100000
    MANIPULATION_RATE = 0.10
    OUTPUT_PATH = 'data/synthetic_labeled/mifid_synthetic.parquet'
    
    # Generate
    generator = SyntheticMiFIDGenerator(
        num_transactions=NUM_TRANSACTIONS,
        manipulation_rate=MANIPULATION_RATE
    )
    
    df = generator.generate()
    
    # Save
    generator.save(df, OUTPUT_PATH)
    
    # Display sample
    print("\nSample Normal Transactions:")
    print(df[df['is_manipulation'] == False].head(3))
    
    print("\nSample Manipulation Transactions:")
    print(df[df['is_manipulation'] == True].head(3))
    
    print("\nColumn Names:")
    print(df.columns.tolist())
    
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nGeneration complete!")


if __name__ == "__main__":
    main()