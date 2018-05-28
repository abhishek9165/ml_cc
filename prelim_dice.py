import tensorflow as tf
g = tf.Graph()
with g.as_default():
  with tf.Session() as sess:
     sess.run(tf.global_variables_initializer())
     a = tf.random_uniform([10,1],minval = 1, maxval = 7, dtype = tf.int32)
     b = tf.random_uniform([10,1],minval = 1, maxval = 7, dtype = tf.int32)
     c = tf.add(a,b)
     d= tf.concat([a,b,c], axis =1)
     sess.run(d)
     # print ([a.eval(),b.eval()])
     print (d.eval())
