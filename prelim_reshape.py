import tensorflow as tf
with tf.Graph().as_default(), tf.Session() as sess:
  # Task: Reshape two tensors in order to multiply them

  # Here are the original operands, which are incompatible
  # for matrix multiplication:
  a = tf.constant([5, 3, 2, 7, 1, 4])
  b = tf.constant([4, 6, 3])
  # We need to reshape at least one of these operands so that
  # the number of columns in the first operand equals the number
  # of rows in the second operand.

  reshaped_a = tf.reshape(a, [6,1])
  reshaped_b = tf.reshape(b, [1,3])

  c = tf.matmul(reshaped_a, reshaped_b)
  #print(reshaped_b.eval())
  print(c.eval())
