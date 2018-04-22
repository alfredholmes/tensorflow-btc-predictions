import tensorflow as tf
import csv, random

#Constants
start = 1100
prediction_length = 500
#neural network takes data from the last input_length days
input_length = 10
#and gives output for the next output_length days
output_length = 1
#This is a first approximation, so just a standard neural network will be used, even though a RNN would be better
#network has hidden_nodes nodes in the hidden layers and hidden_layers hidden layers
hidden_nodes = 10
n_hidden_layers = 4

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
    min = max = None
    for price in initial_data:
        if min is None or min > price:
            min = price
        if max is None or max < price:
            max = price

    min_max = [min, max]
    for i in range(len(initial_data)):
        initial_data[i] = (initial_data[i] - min) / (max - min)

    return data, [initial_data], min_max

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
    price_data, input_, min_max = get_data()

    [min, max] = min_max
    prices = price_data[start:start+input_length]

    price_prediction = []

    for i in range(prediction_length - input_length):
        result = sess.run(pred, feed_dict={input: input_})[0][0]
        price = (result - .33) * 3 * (max - min) + min
        price_prediction.append(price)
        prices.append(price)

        prices = prices[1:]
        print('Price:' + str(price) + 'Last entry:' + str(prices[9]))


        min = max = None
        for p in prices:
            if min is None or min > p:
                min = p
            if max is None or max < p:
                max = p


        for i in range(len(input_[0])):
            input_[0][i] = (prices[i] - min) / (max - min)


    with open('extrapolation.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(price_data[start+input_length:start+prediction_length])
        writer.writerow(price_prediction)








#
