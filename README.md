# Real-Time Market Abuse Detection System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![AWS](https://img.shields.io/badge/AWS-Serverless-orange.svg)](https://aws.amazon.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Production-grade serverless market abuse detection system addressing the real-world challenge of unlabeled transaction data through hybrid ML approach.**

## 🎯 Project Overview

This project demonstrates enterprise-grade ML engineering by building a **real-time market abuse surveillance system** that solves the fundamental challenge every financial institution faces: **detecting manipulation in unlabeled transaction data**.

### The Core Challenge

Financial institutions receive millions of MiFID II transaction reports daily with **no labels** indicating which transactions are manipulative. Traditional approaches fail:
- **Rule-based systems**: 99.99% false positive rate (PwC 2019)
- **Pure unsupervised learning**: 70-80% recall with 20%+ false positives
- **Manual review**: Impossible at scale (30M reports/day to UK FCA alone)

### Our Hybrid Solution

**Three-Phase Approach:**
1. **Synthetic Training**: Generate labeled MiFID II data with known manipulation patterns → Train supervised models (XGBoost)
2. **Transfer Learning**: Apply synthetic-trained models to real unlabeled data → Detect known patterns
3. **Anomaly Detection**: Train LSTM-Autoencoder on real normal transactions → Catch novel manipulation schemes
4. **Active Learning**: Expert review → Manual labeling → Iterative model improvement

### Target Performance

| Metric | Synthetic Test Set | Real Data (Estimated) | Production Goal |
|--------|-------------------|----------------------|-----------------|
| **Throughput** | - | - | **10,000 TPS** |
| **Latency (P99)** | - | - | **<15ms** |
| **Recall** | 95-98% | 80-92% (after active learning) | 85%+ |
| **Cost** | - | - | **<$100/month** |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA GENERATION LAYER                        │
│  Synthetic MiFID II Generator → 100K Labeled Transactions       │
│  Real Transaction Data → Schema Mapping → Unlabeled Inference   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    REAL-TIME PROCESSING LAYER                    │
│  Lambda Producer → Kinesis (3 shards) → Lambda Consumer         │
│                                                                   │
│  Dual Model Inference:                                          │
│  ├─ XGBoost (Synthetic-trained) → Known Patterns               │
│  └─ LSTM-Autoencoder (Real-trained) → Novel Anomalies          │
│                                                                   │
│  Hybrid Ensemble → SHAP Explainability → Flagged Cases         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      STORAGE & ANALYTICS                         │
│  DynamoDB (Real-time) | S3 (Data Lake) | Athena (Analytics)    │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Infrastructure (Serverless AWS):**
- **Kinesis Data Streams**: 3 shards = 10K TPS capacity
- **Lambda (Python 3.12)**: Container images for model inference
- **DynamoDB**: Real-time fraud score storage (<5ms lookups)
- **S3**: Data lake for transactions, models, training data
- **CloudFormation**: Infrastructure as Code

**ML Stack:**
- **XGBoost**: Supervised learning on synthetic labeled data
- **PyTorch LSTM-Autoencoder**: Unsupervised anomaly detection on real data
- **SHAP**: Model explainability (regulatory compliance)
- **scikit-learn**: Feature engineering and preprocessing

**DevOps:**
- **GitHub Actions**: CI/CD pipeline *(planned)*
- **CloudWatch + X-Ray**: Monitoring and distributed tracing
- **Docker**: Containerized Lambda functions
- **pytest**: Comprehensive test coverage

---

## 📁 Current Project Status

### ✅ Completed (Week 1-2)

**Infrastructure Foundation:**
- [x] CloudFormation templates for Kinesis, S3, DynamoDB, IAM
- [x] Deployment scripts with environment-specific configurations
- [x] Python 3.12 virtual environment with all dependencies

**Synthetic Data Generation:**
- [x] Complete MiFID II synthetic data generator (364 lines)
- [x] 100,000 labeled transactions with 10% manipulation patterns
- [x] Three manipulation types: Spoofing, Wash Trading, Layering
- [x] 25+ MiFID II compliant fields (ISIN, LEI, timestamps, prices, etc.)
- [x] Generated dataset: `data/synthetic_labeled/mifid_synthetic.parquet`

**Lambda Producer:**
- [x] Producer Lambda function (191 lines)
- [x] S3 event trigger support
- [x] Batch processing with configurable sizes
- [x] Full dependency package built (`build/producer/`)

**Project Structure:**
```
rt-fraud-detection/
├── cloudformation/           # IaC templates
│   ├── 01-streaming-foundation.yaml
│   ├── 02-lambda-producer.yaml
│   └── config/dev-params.yaml
├── data-generation/
│   └── synthetic_mifid_generator.py
├── data/
│   ├── synthetic_labeled/   # ✅ 100K transactions generated
│   └── real_unlabeled/      # Ready for real data
├── src/
│   ├── producer/            # ✅ Lambda function implemented
│   ├── consumer/
│   ├── feature_engineering/
│   └── models/
│       └── xgboost/
├── scripts/                  # Deployment automation
└── requirements/             # Dependency management
```

### 🚧 In Progress

- [ ] Real data audit and schema mapping workflow
- [ ] Feature engineering (30+ extractors)
- [ ] XGBoost training pipeline
- [ ] End-to-end throughput testing (10K TPS)

### 📋 Roadmap

**Week 2: Model Training**
- [ ] Implement 30+ feature extractors
- [ ] Train XGBoost on synthetic data (target: 95%+ recall)
- [ ] Deploy XGBoost as Lambda container
- [ ] Add SHAP explainability

**Week 3: Transfer Learning & Anomaly Detection**
- [ ] Apply synthetic-trained model to real data
- [ ] Train LSTM-Autoencoder on real normal transactions
- [ ] Build hybrid ensemble detector
- [ ] Flag suspicious cases for review

**Week 4: Active Learning & Production**
- [ ] Manual review interface
- [ ] Model retraining with real labels
- [ ] CloudWatch monitoring dashboards
- [ ] CI/CD pipeline
- [ ] Production documentation

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+**
- **AWS Account** with CLI configured
- **AWS Permissions**: Kinesis, Lambda, S3, DynamoDB, IAM, CloudFormation
- **Docker** (for Lambda container builds)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/JosianQuintanaArroyoKaizen/rt-fraud-detection.git
cd rt-fraud-detection
```

2. **Set up Python environment:**
```bash
# Create virtual environment
python3.12 -m venv rt-anomaly-detection
source rt-anomaly-detection/bin/activate  # Linux/Mac
# rt-anomaly-detection\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements/base.txt
```

3. **Generate synthetic training data:**
```bash
python data-generation/synthetic_mifid_generator.py
```

Expected output:
```
Total Transactions: 100,000
Normal: 90,000
Manipulation: 10,000
  - SPOOFING: 5,000
  - WASH_TRADING: 3,000
  - LAYERING: 2,000
```

4. **Deploy infrastructure:**
```bash
cd cloudformation

# Generate unique suffix for resource names
./generate-suffix.sh

# Deploy streaming foundation (Kinesis + S3 + DynamoDB)
./deploy-streaming.sh dev

# Deploy Lambda producer
cd ../scripts
./deploy-producer.sh dev
```

5. **Test the producer:**
```bash
chmod +x ./scripts/test-producer.sh
./scripts/test-producer.sh dev
```

---

## 📊 Synthetic Data Schema

Our generator produces MiFID II-compliant transaction records with 25+ fields:

### Core Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `transaction_id` | string | Unique identifier | UUID |
| `timestamp` | datetime | Execution time (microsecond precision) | 2025-10-20T14:23:45.123456 |
| `instrument_id` | string | ISIN code | GB00B1YW4409 |
| `trader_id` | string | LEI identifier | LEI549300... |
| `price` | float | Execution price | 125.47 |
| `quantity` | int | Transaction size | 1000 |
| `side` | enum | BUY or SELL | BUY |
| `trading_venue` | string | Exchange MIC code | XLON |

### Manipulation Labels

| Field | Type | Description |
|-------|------|-------------|
| `is_manipulation` | boolean | Ground truth label |
| `manipulation_type` | enum | SPOOFING, WASH_TRADING, LAYERING, NONE |

### Market Context

| Field | Type | Description |
|-------|------|-------------|
| `bid_price` | float | Best bid at execution |
| `ask_price` | float | Best ask at execution |
| `bid_size` | int | Depth at best bid |
| `ask_size` | int | Depth at best ask |

**Full schema:** See [`data-generation/synthetic_mifid_generator.py`](data-generation/synthetic_mifid_generator.py)

---

## 🧪 Manipulation Patterns

Our synthetic generator implements three common market abuse schemes:

### 1. Spoofing (5% of data)
```
Pattern: Large fake orders → Quick cancellation → Real execution opposite side
Detection Features:
  - High order_cancel_ratio (>0.8)
  - Rapid order lifecycle (<5 seconds)
  - Price impact on opposite side
```

### 2. Wash Trading (3% of data)
```
Pattern: Matched buy/sell between controlled accounts
Detection Features:
  - High counterparty_relationship_score
  - Same price, size, timestamp
  - No economic purpose
```

### 3. Layering (2% of data)
```
Pattern: Multiple orders at different price levels → All cancelled except one
Detection Features:
  - High order_to_trade_ratio
  - Sequential price levels
  - Coordinated timing
```

---

## 🔧 Configuration

### Environment Variables

**Lambda Producer:**
```bash
KINESIS_STREAM_NAME=dev-rtfd-stream-abc123
BATCH_SIZE=100
MAX_RECORDS=500
```

**Infrastructure Parameters:**
```yaml
# cloudformation/config/dev-params.yaml
Environment: dev
ProjectName: rtfd
KinesisShardCount: 3        # 10K TPS capacity
KinesisRetentionHours: 24
```

### Cost Optimization

Current configuration targets **<$100/month** for 10K TPS:

| Service | Cost/Month | Notes |
|---------|-----------|-------|
| Kinesis (3 shards) | ~$32 | 730 shard-hours |
| Lambda | ~$50 | ~259M invocations |
| DynamoDB | ~$6 | On-demand pricing |
| S3 | ~$3 | Data storage |
| CloudWatch | ~$5 | Logs and metrics |
| **Total** | **~$96** | Serverless scales to zero |

---

## 🧑‍💻 Development

### Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# With coverage
pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ data-generation/ tests/

# Lint
flake8 src/ data-generation/

# Type checking
mypy src/
```

### Local Development

```bash
# Activate environment
source activate-env.sh

# Run synthetic generator
python data-generation/synthetic_mifid_generator.py

# Test Lambda locally (requires SAM CLI)
sam local invoke ProducerFunction --event test-events/sample.json
```

---

## 📈 Performance Benchmarks

### Current Status (Week 1-2)

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Synthetic Data | 100K records | ✅ 100K | Complete |
| Manipulation Rate | 10% | ✅ 10% | Complete |
| Data Quality | MiFID II compliant | ✅ 25+ fields | Complete |
| Infrastructure | Deployed | 🚧 Pending test | In Progress |
| Throughput | 10K TPS | ⏳ Not measured | Next |

### Expected Performance (After Week 4)

**Model Performance:**
- Synthetic test set: **95-98% recall**, **92-96% precision**
- Real data (initial): **70-85% estimated recall**
- Real data (after active learning): **85-92% estimated recall**

**System Performance:**
- Throughput: **10,000 TPS sustained**
- Latency (P99): **<15ms end-to-end**
- Availability: **99.9%** (serverless)

---

## 🎓 Why This Approach Matters

### Real-World Problem Solving

Unlike academic projects using pre-labeled Kaggle datasets, this demonstrates how financial institutions **actually** build surveillance systems:

1. **No Ground Truth**: MiFID II reports arrive unlabeled
2. **Synthetic Bootstrapping**: Generate labeled data to learn patterns
3. **Transfer Learning**: Apply knowledge to real transactions
4. **Anomaly Detection**: Catch novel schemes not in training data
5. **Active Learning**: Continuously improve with expert feedback

### Production Engineering

This project showcases:
- ✅ **Serverless Architecture**: Auto-scaling, cost-efficient
- ✅ **Infrastructure as Code**: Reproducible deployments
- ✅ **ML Pipeline Engineering**: Training, inference, monitoring
- ✅ **Regulatory Compliance**: SHAP explainability for MAR/MiFID II
- ✅ **Hybrid ML Strategy**: Supervised + Unsupervised + Active Learning

---

## 📚 Documentation

- **[Build Guide](Real-Time%20Market%20Abuse%20Detection%20System%20-%20Build%20Guide.md)**: Complete 4-week implementation plan
- **[Architecture](#)**: Detailed system design *(coming soon)*
- **[Performance](docs/PERFORMANCE.md)**: Benchmarks and metrics *(coming soon)*
- **[API Documentation](docs/api/)**: Lambda function interfaces *(coming soon)*

---

## 🤝 Contributing

This is a portfolio/demonstration project. Contributions welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🔗 References

### AWS Services
- [AWS Lambda Python 3.12 Runtime](https://docs.aws.amazon.com/lambda/latest/dg/lambda-python.html)
- [Kinesis Data Streams](https://docs.aws.amazon.com/kinesis/latest/dev/introduction.html)
- [Lambda Container Images](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html)

### Regulatory Context
- [FCA Market Watch 79](https://www.fca.org.uk/publications/market-watch)
- [ESMA MiFID II Transaction Reporting](https://www.esma.europa.eu/data-reporting/mifir-reporting)

### ML Techniques
- Transfer Learning in Finance
- Active Learning Strategies
- Anomaly Detection with Autoencoders

---

## 👤 Author

**Josian Quintana Arroyo**

- GitHub: [@JosianQuintanaArroyoKaizen](https://github.com/JosianQuintanaArroyoKaizen)
- Project: Real-Time Market Abuse Detection System
- Focus: Production ML Engineering for Financial Surveillance

---

## 🏆 Project Goals

- [x] Address real-world unlabeled data challenge
- [x] Generate 100K labeled synthetic MiFID II transactions
- [x] Build serverless infrastructure (Kinesis, Lambda, S3, DynamoDB)
- [ ] Achieve 10,000 TPS throughput with <15ms latency
- [ ] Train XGBoost on synthetic data (95%+ recall)
- [ ] Apply transfer learning to real unlabeled data
- [ ] Implement LSTM-Autoencoder for anomaly detection
- [ ] Build active learning pipeline
- [ ] Deploy production monitoring and CI/CD
- [ ] Create comprehensive technical documentation

**Target Completion:** 4 weeks from October 20, 2025

---

*Last Updated: October 20, 2025*
