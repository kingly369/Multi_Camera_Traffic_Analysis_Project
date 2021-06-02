import pika, sys, os, base64

# Makes a connection to the other jetson
def makeConnection():
    credential = pika.PlainCredentials('jack','rMQtest')
    connection = pika.BlockingConnection(pika.ConnectionParameters('192.168.0.32', 5672, 'custom-vhost', credential))
    channel = connection.channel()

    status = channel.queue_declare(queue='KentQ')
    status2 = channel.queue_declare(queue='JackQ')

    def callback(ch, method, properties, body):
        print(" [x] Received %r" % body.decode())
    print("Kent jetson has {} messages".format(status.method.message_count))
    print("Jackson jetson has {} messages".format(status2.method.message_count))
    print(' [*] Established Connection. To exit press CTRL+C')
    return channel

# To initiate communication with the other jetsons
def startupjack():
    channel = makeConnection()
    for method_frame, properties, body in channel.consume('JackQ'):
        startSTR = body.decode('ascii')
        print('startstr', startSTR)

        channel.basic_ack(method_frame.delivery_tag)
        if (startSTR == 'startinference'):
            print('startupreceived on worker')
            return channel
    return channel

# Checks the Rabbit Queue if there's any messages
def consumeMessage(channel):
    counter = 0
    message = []
    for method_frame, properties, body in channel.consume('JackQ'):
        message.append(body.decode('ascii')) #.decode'ascii'
        channel.basic_ack(method_frame.delivery_tag)
        counter += 1
        if counter == 1:
            break
    channel.cancel()
    print('Stopped consuming')
    return message

def clearMessage(channel):
    counter = 0
    for method_frame, properties, body in channel.consume('JackQ'):
        print(body.decode())
        channel.basic_ack(method_frame.delivery_tag)
        status = channel.queue_declare(queue='JackQ')
        if status.method.message_count == 0:
            print("In if statement")
            break
    channel.cancel()
    print('Stopped consuming')

def sendMessage(channel, message):
    channel.basic_publish(exchange='', routing_key='KentQ', body=message)

if __name__ == '__main__':
    try:
        channel = makeConnection()
        consumeMessage(channel)
        consumeMessage(channel)
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
