from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from controller import Controller

INPUT_SIZE = 50
NUM_FEATURES = 4
LABEL_SIZE = 1
UNITS = 64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

controller = Controller()

@app.get("/")
def root():
    return "Welcome to BUF project!"

@app.get("/exchange")
def getExchangeList():
    return {"exchanges": controller.getExchangeList(), "status": status.HTTP_200_OK}

@app.get("/stock")
def getStockList(exchange: str = Query("HNX")):
    exchanges = controller.getExchangeList()
    if exchange not in exchanges:
        return {"message": "Invalid Exchange", "status": status.HTTP_404_NOT_FOUND}
    return {"stocks": controller.getStockList(exchange), "status": status.HTTP_200_OK}

@app.get("/record")
def getStockRecords(stock: str = Query("AAV"),
             exchange: str = Query("HNX")):
    documents = controller.load(exchange, stock)
    if documents is None:
        return {"message": "Invalid stock", "status": status.HTTP_404_NOT_FOUND}
    return documents, status.HTTP_200_OK


@app.get("/train")
def train(inputSize: int = Query(INPUT_SIZE),
        numFeatures: int = Query(NUM_FEATURES),
        units: int = Query(UNITS),
        optimizer: str = Query("adam"),
        epochs: int = Query(100),
        batchSize: int = Query(64),
        validationSplit: float = Query(0.1)):
    controller.train(inputSize, numFeatures, units, optimizer, epochs, batchSize, validationSplit)
    return {"message": "Complete!", "status": status.HTTP_200_OK}

@app.get("/evaluation")
def evaluation():
    evaResult = controller.evaluation()
    return {"result": evaResult, "status": status.HTTP_200_OK}

@app.get("/predict")
def predict(exchange: str = Query("HNX"),
            stock: str = Query("AAV"),
            date: str = Query("02/14/2024")):
    try:
        prediction = controller.predict(stock, exchange, date)
        if prediction is None:
            return {"message": "Invalid stock", "status": status.HTTP_404_NOT_FOUND}
        return prediction, status.HTTP_200_OK
    except HTTPException as e:
        return {"message": e.detail, "status": e.status_code}

@app.get("*")
def error():
    return {"message": "Invalid API", "status": status.HTTP_404_NOT_FOUND}