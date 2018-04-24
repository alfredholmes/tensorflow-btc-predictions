import tensorflow as tf
import csv, random

#Constants

#neural network takes data from the last input_length days
input_length = 10
#and gives output for the next output_length days
output_length = 1
#This is a first approximation, so just a standard neural network will be used, even though a RNN would be better
#network has hidden_nodes nodes in the hidden layers and hidden_layers hidden layers
hidden_nodes = 30
n_hidden_layers = 4

learning_rate = 0.001
print(random.randint(0, 10))
#function to get data
#output.csv has daily trading data with data in the form [date, price, volume] per row
def get_data():
    data = []
    with open('output.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(float(row[1]))


    training_input = []
    training_output = []
    for i in range(0, 10000):
        offset = random.randint(0, 700)
        a = data[offset:offset + input_length]
        b = data[offset+input_length:offset+input_length + output_length]

        min = max = None

        for d in a:
            if min is None or d < min:
                min = d
            if max is None or d > max:
                max = d

        for j in range(len(a)):
            a[j] = (a[j] - min) / ((max - min))
        for j in range(len(b)):
            b[j] = 0.33 + (b[j] - min ) / (3 * (max - min))

        training_input.append(a)
        training_output.append(b)
    testing_input = []
    testing_output = []
    offset = 700
    testing_min_max = []
    while offset < 1179:
        #offset = random.randint(707, 1185)
        a = data[offset:offset + input_length]
        b = data[offset+input_length:offset+input_length + output_length]

        min = max = None

        for d in a:
            if min is None or d < min:
                min = d
            if max is None or d > max:
                max = d


        for i in range(len(a)):
            a[i] = (a[i] - min) / ((max - min))
        for i in range(len(b)):
            b[i] = 0.33 + (b[i] - min) / (3 * (max - min))

        testing_input.append(a)
        testing_output.append(b)
        testing_min_max.append([min, max])
        #print(a)
        #print(b)
        offset += 1
    return training_input, training_output, testing_input, testing_output, testing_min_max

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

cost = tf.reduce_mean(tf.square(error))

training_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#save the graph
saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, "./save/network")
    input_, output_,test_input_, test_output_, min_max = get_data()

    for i in range(1000):
        sess.run(training_step, feed_dict={input: input_, output: output_})
        print(sess.run(cost, feed_dict={input: test_input_, output: test_output_}))

    saver.save(sess, './save/network')
    result = sess.run(pred, feed_dict={input: test_input_, output: test_output_})
    error = sess.run(error, feed_dict={input:test_input_, output:test_output_})
    with open('nn_predictions.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)

        actual = []
        for i in range(len(test_output_)):

            [min, max] = min_max[i]
            for j in range(len(test_output_[i])):
                actual.append((test_output_[i][j] - 0.33) * 3 * (max - min) + min)
        p = []
        e = []
        for i in range(len(result)):
            d = test_input_[i]
            [min, max] = min_max[i]
            for j in range(len(result[i])):
                p.append((result[i][j] - 0.33) * 3 * (max - min) + min)
            for j in range(len(error[i])):
                e.append(error[i][j]**2)

        writer.writerow(actual)
        writer.writerow(p)
        writer.writerow(e)








#
