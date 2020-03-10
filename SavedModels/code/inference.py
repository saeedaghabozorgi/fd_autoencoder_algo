import base64
import io
import json
import requests
import tensorflow as tf
from google.protobuf.json_format import MessageToDict
from string import whitespace

def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API
    Args:
        data (obj): the request data stream
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    if context.request_content_type == 'application/x-tfexample':
        payload = data.read()
        example = tf.train.Example()
        example.ParseFromString(payload)
        example_feature = MessageToDict(example.features)['feature']
        encoded_image = example_feature['x']['floatList']['value']
        instance = [encoded_image]
        return json.dumps({"instances": instance})

        
    elif context.request_content_type == 'application/x-recordio-protobuf':
        payload = data.read()
        encoded_image = base64.b64encode(payload).decode('utf-8')
        instance = [{"b64": encoded_image}]
        return json.dumps({"instances": instance})
    elif context.request_content_type == 'application/x-image':
        payload = data.read()
        encoded_image = base64.b64encode(payload).decode('utf-8')
        instance = [{"b64": encoded_image}]
        return json.dumps({"instances": instance})
    else:
        _return_error(415, 'Unsupported content type  "{}"'.format(context.request_content_type or 'Unknown'))


def output_handler(response, context):
    """Post-process TensorFlow Serving output before it is returned to the client.
    Args:
        response (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """

    if response.status_code != 200:
        _return_error(response.status_code, response.content.decode('utf-8'))

    response_content_type = context.accept_header
    prediction = response.content
    return prediction, response_content_type


def _return_error(code, message):
    raise ValueError('Error: {}, {}'.format(str(code), message))