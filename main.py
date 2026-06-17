import json, pickle
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel

INPUT_DIM = 43


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1     = nn.Linear(INPUT_DIM, 128)
        self.fc2     = nn.Linear(128, 64)
        self.fc3     = nn.Linear(64, 32)       
        self.fc4     = nn.Linear(32, 1)        
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.bn1     = nn.BatchNorm1d(128)
        self.bn2     = nn.BatchNorm1d(64)
        self.bn3     = nn.BatchNorm1d(32)      

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))  
        x = self.dropout(x)
        x = self.fc4(x)                        
        return x

# Load config — includes threshold
with open('artifacts/model_config.json') as f:
    cfg = json.load(f)

THRESHOLD = cfg.get('threshold', 0.5)


# Load model
model = MLP()
model.load_state_dict(torch.load('artifacts/best_model.pt', map_location='cpu'))
model.eval()

# Load scaler
with open('artifacts/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = FastAPI(title='Credit Default Prediction API')

class CustomerFeatures(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

class PredictionResponse(BaseModel):
    default_probability: float
    prediction: int
    label: str
    threshold_used: float

@app.get('/health')
def health():
    return {'status': 'healthy'}

@app.post('/predict', response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    # Original features
    LIMIT_BAL = customer.LIMIT_BAL
    PAY_vals  = [customer.PAY_0, customer.PAY_2, customer.PAY_3,
                 customer.PAY_4, customer.PAY_5, customer.PAY_6]
    BILL_vals = [customer.BILL_AMT1, customer.BILL_AMT2, customer.BILL_AMT3,
                 customer.BILL_AMT4, customer.BILL_AMT5, customer.BILL_AMT6]
    PAY_AMTS  = [customer.PAY_AMT1, customer.PAY_AMT2, customer.PAY_AMT3,
                 customer.PAY_AMT4, customer.PAY_AMT5, customer.PAY_AMT6]

    # Engineered features
    TOTAL_DELAY        = sum(p > 0 for p in PAY_vals)
    MAX_DELAY          = max(PAY_vals)
    CURRENTLY_DELAYED  = int(customer.PAY_0 > 0)
    CONSECUTIVE_DELAY  = int(customer.PAY_0>0) + int(customer.PAY_2>0) + int(customer.PAY_3>0)

    AVG_BILL           = sum(BILL_vals) / 6
    UTILIZATION        = AVG_BILL / (LIMIT_BAL + 1)
    BILL_TREND         = customer.BILL_AMT1 - customer.BILL_AMT6
    MAX_BILL           = max(BILL_vals)

    TOTAL_PAID         = sum(PAY_AMTS)
    AVG_PAID           = TOTAL_PAID / 6
    PAY_TREND          = customer.PAY_AMT1 - customer.PAY_AMT6
    ZERO_PAY_COUNT     = sum(p == 0 for p in PAY_AMTS)

    BAL1               = customer.BILL_AMT1 - customer.PAY_AMT1
    BAL2               = customer.BILL_AMT2 - customer.PAY_AMT2
    BAL3               = customer.BILL_AMT3 - customer.PAY_AMT3
    AVG_REMAINING_BAL  = (BAL1 + BAL2 + BAL3) / 3
    BAL_GROWING        = int(BAL1 > BAL3)

    AGE                = customer.AGE
    AGE_GROUP          = 1 if AGE<=25 else 2 if AGE<=35 else 3 if AGE<=45 else 4 if AGE<=60 else 5
    HIGH_EDUCATION     = int(customer.EDUCATION <= 2)
    LIMIT_PER_AGE      = LIMIT_BAL / AGE

    features = np.array([[
        # Original 23 features
        customer.LIMIT_BAL, customer.SEX,       customer.EDUCATION,
        customer.MARRIAGE,  customer.AGE,
        customer.PAY_0,     customer.PAY_2,      customer.PAY_3,
        customer.PAY_4,     customer.PAY_5,      customer.PAY_6,
        customer.BILL_AMT1, customer.BILL_AMT2,  customer.BILL_AMT3,
        customer.BILL_AMT4, customer.BILL_AMT5,  customer.BILL_AMT6,
        customer.PAY_AMT1,  customer.PAY_AMT2,   customer.PAY_AMT3,
        customer.PAY_AMT4,  customer.PAY_AMT5,   customer.PAY_AMT6,
        # Engineered features
        TOTAL_DELAY, MAX_DELAY, CURRENTLY_DELAYED, CONSECUTIVE_DELAY,
        AVG_BILL, UTILIZATION, BILL_TREND, MAX_BILL,
        TOTAL_PAID, AVG_PAID, PAY_TREND, ZERO_PAY_COUNT,
        BAL1, BAL2, BAL3, AVG_REMAINING_BAL, BAL_GROWING,
        AGE_GROUP, HIGH_EDUCATION, LIMIT_PER_AGE
    ]], dtype=np.float32)

    scaled = scaler.transform(features)
    tensor = torch.tensor(scaled, dtype=torch.float32)

    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()

    pred  = int(prob > THRESHOLD)
    label = 'Will Default' if pred == 1 else 'Will Not Default'
    return PredictionResponse(
        default_probability=round(prob, 4),
        prediction=pred,
        label=label,
        threshold_used=THRESHOLD
    )