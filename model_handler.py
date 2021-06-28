from ts.torch_handler.base_handler import BaseHandler
import transformers
from torch.nn.utils.rnn import pad_sequence
import torch
import json
import os
import re
import logging
import time

logger = logging.getLogger(__name__)

class SentimentalHandler(BaseHandler):

    def __init__(self):
        super(SentimentalHandler, self).__init__()

    def initialize(self, context):
        super(SentimentalHandler, self).initialize(context)
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        # read configs for the max_length from setup_config.json
        setup_config_path = os.path.join(model_dir, "setup_config.json")
        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                self.setup_config = json.load(setup_config_file)
        else:
            logger.warning('Missing the setup_config.json file.')


        self.tokenizer = transformers.CamembertTokenizer.from_pretrained("camembert-base")
        #self.model.to(self.device)
        self.model.eval()
        self.initialized = True

    def text_cleaner(self, text):
        # Substituting multiple spaces with single space
        text = re.sub(r'\s+', ' ', text, flags=re.I)
        # Converting to Lowercase
        text = text.lower()
        return text
    
    def preprocess(self, data):
        textInput = []
        rawText  = []
        for row in data:
            text = row.get("data") or row.get("body") or row.get("sentence")
            text = text.decode('utf-8')
            # cleaning
            text = self.text_cleaner(text)
            encoded_text = self.tokenizer.encode(text, 
                                                 max_length=self.setup_config['max_length'], 
                                                 truncation=True)
            encoded_text = torch.tensor(encoded_text)
            textInput.append(encoded_text)
            rawText.append(text)
        # stack to build one tensor with batch x length
        prep_text = pad_sequence(textInput, batch_first=True)
        return prep_text, rawText

    def inference(self, data, *args, **kwargs):
        with torch.no_grad():
            model_output = self.model(data)
            probas = torch.softmax(model_output.logits, 1)
            values, indices = torch.topk(probas, 1)
            category = indices.detach().cpu()
            batch_size = len(category)
            category = category.reshape(batch_size)
            category = category.numpy().tolist()
            
            values = values.reshape(batch_size)
            values = values.numpy().tolist()
            
        logger.info("Model predicted: %s", category)
        return category, values

    def postprocess(self, data, raw_data, probas):
        postp_output = []
        for i in range(0, len(data)):
            
            output = {
                "input": raw_data[i],
                "sentiment": "positive" if data[i] == 1 else "negative",
                "confidence": str(probas[i])
            }
            postp_output.append(json.dumps(output))
        return postp_output
    
    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediction output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        start_time = time.time()
        
        model_input, raw_data = self.preprocess(data)
        categories, probas = self.inference(model_input)
        
        stop_time = time.time()
        duration = round((stop_time - start_time) * 1000, 2)
        context.metrics.add_time('HandlerTime', duration, None, 'ms')
        context.metrics.add_time('MeanTimePerRequest', duration / len(data), None, 'ms')
        return self.postprocess(categories, raw_data, probas)