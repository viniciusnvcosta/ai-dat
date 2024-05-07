import json
from magnum import Magnum

def lambda_handler(event, context):
    # Extract data from the event
    # e.g. event['key']

    # Perform your logic here
    # e.g. result = my_function(event['key'])

    # Initialize Magnum
    magnum = Magnum()

    # Process the event using Magnum
    result = magnum.process(event)

    # Prepare the response
    response = {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Hello from AWS Lambda!',
            'result': result
        })
    }

    return response