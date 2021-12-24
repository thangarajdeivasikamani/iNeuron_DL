from utils.model import Perceptron
from utils.all_utils import prepare_data,save_plot,save_model
import pandas as pd
import logging
import os
from tqdm import tqdm

logging_str ="[%(asctime)s:%(levelname)s:%(module)s:%(message)s"
log_dir ="logs"
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(filename = os.path.join(log_dir,"running_log.log"),level=logging.INFO,format=logging_str,filemode="a")


def main(data,eta,epochs,model_filename,plot_filename):
 
    df = pd.DataFrame(data)
    logging.info(f"this is actual dataframe{df}")
    X,y = prepare_data(df)

    

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()
    
    save_model(model, filename=model_filename)
    save_plot(df, plot_filename, model)
   

if __name__=='__main__':  ## Entry point
  
    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,0,0,1],
    }
    ETA = 0.3 # 0 and 1
    EPOCHS = 10
    
    try:
        logging.info(">>>\nStarting the Traninig>>>")
        main(data=AND,eta=ETA,epochs=EPOCHS,model_filename="and.model",plot_filename="and.png")
        logging.info("<<<End the Traninig<<<")

    except Exception as e:
        logging.exception(e)
        raise e

    