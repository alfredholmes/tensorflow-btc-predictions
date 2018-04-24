import tensorflow as tf
import csv, random

#Constants
start = 800
prediction_length = 100
#neural network takes data from the last input_length days
input_length = 10
#and gives output for the next output_length days
output_length = 1
#This is a first approximation, so just a standard neural network will be used, even though a RNN would be better
#network has hidden_nodes nodes in the hidden layers and hidden_layers hidden layers
hidden_nodes = 30
n_hidden_layers = 4

e = 0.007
#sd = 0.005

#scale the prices in data in the
def scale(data, s = 1):
    min = max = None
    for d in data:
        if min is None or min > d:
            min = d
        if max is None or max < d:
            max = d
    a = []
    if min != max:
        for d in data:
            a.append((d - min) / (s * (max - min)) + (1.0 - 1.0 / s) / 2.0)
    else:
        for d in data:
            a.append(0.5)
    return a, min, max

def unscale(data, min, max, s = 1):
    a = []
    for d in data:
        a.append((d - 0.33) * 3 * (max - min) + min)

    return a

print(random.randint(0, 10))
#function to get data
#output.csv has daily trading data with data in the form [date, price, volume] per row
def get_data():
    data = []
    with open('output.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(float(row[1]))

    initial_data = data[start:start+input_length]
    initial_data, min, max = scale(initial_data)

    return data, [initial_data], min, max

#defining tensorflow variables
input = tf.placeholder(tf.float32, [None, input_length])
output = tf.placeholder(tf.float32, [None, output_length])
test_input = tf.placeholder(tf.float32, [None, input_length])
test_output = tf.placeholder(tf.float32, [None, output_length])


hidden_layers = [tf.Variable(tf.random_uniform([input_length, hidden_nodes], -1, 1))]
for i in range(1, n_hidden_layers-1):
    tf.Variable(tf.random_uniform([hidden_nodes, hidden_nodes], -1, 1))
hidden_layers.append(tf.Variable(tf.random_uniform([hidden_nodes, output_length], -1, 1)))

#define the forward pass
def forwardpass(inp):
    out = inp
    for layer in hidden_layers:
        out = tf.sigmoid(tf.matmul(out, layer))

    return out

pred = forwardpass(input)
error = output - pred



#save the graph
saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./save/network")

    price_data, input_, min, max = get_data()

    with open('extrapolation.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(price_data[start+input_length:start+input_length+prediction_length])

        for j in range(10):
            price_data, input_, min, max = get_data()


            prices = price_data[start:start+input_length]

            price_prediction = []
            print('Run ' + str(j))
            for i in range(prediction_length):
                result = float(sess.run(pred, feed_dict={input: input_})[0][0]) + random.gauss(0, e)

                #result = result + e


                price = unscale([result], min, max, 3)[0]
                #price = unscale([float(result) + random.gauss(0, error)], min, max, 3)[0]

                if price < 0:
                    price = 0

                price_prediction.append(price)
                prices.append(price)


                prices = prices[1:]
                #print('Price:' + str(price) + 'Last entry:' + str(prices[9]))

                input_, min, max = scale(prices, 3)
                input_ = [input_]





            writer.writerow(price_prediction)
            price_prediction = []








#
