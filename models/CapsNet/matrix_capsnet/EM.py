import tensorflow as tf

epsilon = 1e-9


def matrix_capsules_em_routing(votes, i_activations, beta_v, beta_a, iterations, name):
    """The EM routing between input capsules (i) and output capsules (o).

  :param votes: [N, OH, OW, KH x KW x I, O, PH x PW] from capsule_conv(),
    or [N, KH x KW x I, O, PH x PW] from capsule_fc()
  :param i_activation: [N, OH, OW, KH x KW x I, O] from capsule_conv(),
    or [N, KH x KW x I, O] from capsule_fc()
  :param beta_v: [1, 1, 1, O] from capsule_conv(),
    or [1, O] from capsule_fc()
  :param beta_a: [1, 1, 1, O] from capsule_conv(),
    or [1, O] from capsule_fc()
  :param iterations: number of iterations in EM routing, often 3.
  :param name: name.

  :return: (pose, activation) of output capsules.

  note: the comment assumes arguments from capsule_conv(), remove OH, OW if from capsule_fc(),
    the function make sure is applicable to both cases by using negative index in argument axis.
  """
    # votes: [N, OH, OW, KH x KW x I, O, PH x PW]
    votes_shape = votes.get_shape().as_list()
    # i_activations: [N, OH, OW, KH x KW x I]
    i_activations_shape = i_activations.get_shape().as_list()

    with tf.variable_scope(name) as scope:

        # note: match rr shape, i_activations shape with votes shape for broadcasting in EM routing

        # rr: [1, 1, 1, KH x KW x I, O, 1],
        # rr: routing matrix from each input capsule (i) to each output capsule (o)
        rr = tf.constant(1.0 / votes_shape[-2], shape=votes_shape[-3:-1] + [1], dtype=tf.float32)

        # i_activations: expand_dims to [N, OH, OW, KH x KW x I, 1, 1]
        i_activations = i_activations[..., tf.newaxis, tf.newaxis]

        # beta_v and beta_a: expand_dims to [1, 1, 1, 1, O, 1]
        beta_v = beta_v[..., tf.newaxis, :, tf.newaxis]
        beta_a = beta_a[..., tf.newaxis, :, tf.newaxis]

        def m_step(rr, votes, i_activations, beta_v, beta_a, inverse_temperature):
            """The M-Step in EM Routing.

      :param rr: [1, 1, 1, KH x KW x I, O, 1], or [N, KH x KW x I, O, 1],
        routing assignments from each input capsules (i) to each output capsules (o).
      :param votes: [N, OH, OW, KH x KW x I, O, PH x PW], or [N, KH x KW x I, O, PH x PW],
        input capsules poses x view transformation.
      :param i_activations: [N, OH, OW, KH x KW x I, 1, 1], or [N, KH x KW x I, 1, 1],
        input capsules activations, with dimensions expanded to match votes for broadcasting.
      :param beta_v: cost of describing capsules with one variance in each h-th compenents,
        should be learned discriminatively.
      :param beta_a: cost of describing capsules with one mean in across all h-th compenents,
        should be learned discriminatively.
      :param inverse_temperature: lambda, increase over steps with respect to a fixed schedule.

      :return: (o_mean, o_stdv, o_activation)
      """

            # votes: [N, OH, OW, KH x KW x I, O, PH x PW]
            # rr_prime: [N, OH, OW, KH x KW x I, O, 1]
            rr_prime = rr * i_activations
            # rr_prime = rr_prime / (tf.reduce_sum(rr_prime, axis=-2, keep_dims=True) + epsilon)

            # rr_prime_sum: sum acorss i, [N, OH, OW, 1, O, 1]
            rr_prime_sum = tf.reduce_sum(rr_prime, axis=-3, keepdims=True, name='rr_prime_sum')

            # o_mean: [N, OH, OW, 1, O, PH x PW]
            o_mean = tf.reduce_sum(rr_prime * votes, axis=-3, keepdims=True) / rr_prime_sum

            # o_stdv: [N, OH, OW, 1, O, PH x PW]
            o_stdv = tf.sqrt(tf.reduce_sum(rr_prime * tf.square(votes - o_mean), axis=-3, keepdims=True) / rr_prime_sum)

            # o_cost_h: [N, OH, OW, 1, O, PH x PW]
            o_cost_h = (beta_v + tf.log(o_stdv + epsilon)) * rr_prime_sum

            # o_activation: [N, OH, OW, 1, O, 1]
            # o_activations_cost = (beta_a - tf.reduce_sum(o_cost_h, axis=-1, keep_dims=True))
            # yg: This is to stable o_cost, which often large numbers, using an idea like batch norm.
            # It is in fact only the relative variance between each channel determined which one should activate,
            # the `relative` smaller variance, the `relative` higher activation.
            # o_cost: [N, OH, OW, 1, O, 1]
            o_cost = tf.reduce_sum(o_cost_h, axis=-1, keepdims=True)
            o_cost_mean = tf.reduce_mean(o_cost, axis=-2, keepdims=True)
            o_cost_stdv = tf.sqrt(tf.reduce_sum(tf.square(o_cost - o_cost_mean), axis=-2, keepdims=True)
                                  / o_cost.get_shape().as_list()[-2])
            o_activations_cost = beta_a + (o_cost_mean - o_cost) / (o_cost_stdv + epsilon)
            # tf.summary.histogram('o_activation_cost', o_activations_cost)
            o_activations = tf.sigmoid(inverse_temperature * o_activations_cost)
            # tf.summary.histogram('o_activation', o_activations)
            return o_mean, o_stdv, o_activations

        def e_step(o_mean, o_stdv, o_activations, votes):
            """The E-Step in EM Routing.

      :param o_mean: [N, OH, OW, 1, O, PH x PW], or [N, 1, O, PH x PW],
      :param o_stdv: [N, OH, OW, 1, O, PH x PW], or [N, 1, O, PH x PW],
      :param o_activations: [N, OH, OW, 1, O, 1], or [N, 1, O, 1],
      :param votes: [N, OH, OW, KH x KW x I, O, PH x PW], or [N, KH x KW x I, O, PH x PW],

      :return: rr
      """

            # votes: [N, OH, OW, KH x KW x I, O, PH x PW]
            # o_p: [N, OH, OW, KH x KW x I, O, 1]
            # o_p is the probability density of the h-th component of the vote from i to c
            o_p_unit0 = - tf.reduce_sum(
                tf.square(votes - o_mean) / (2 * tf.square(o_stdv)), axis=-1, keepdims=True)

            o_p_unit2 = - tf.reduce_sum(tf.log(o_stdv + epsilon), axis=-1, keepdims=True)

            # o_p
            o_p = o_p_unit0 + o_p_unit2

            # rr: [N, OH, OW, KH x KW x I, O, 1]
            zz = tf.log(o_activations + epsilon) + o_p
            rr = tf.nn.softmax(zz, axis=len(zz.get_shape().as_list()) - 2)
            tf.summary.histogram('rr', rr)
            return rr

        # inverse_temperature (min, max)
        # y=tf.sigmoid(x): y=0.50,0.73,0.88,0.95,0.98,0.99995458 for x=0,1,2,3,4,10,
        it_min = 1.0
        it_max = min(iterations, 3.0)
        for it in xrange(iterations):
            inverse_temperature = it_min + (it_max - it_min) * it / max(1.0, iterations - 1.0)
            o_mean, o_stdv, o_activations = m_step(
                rr, votes, i_activations, beta_v, beta_a, inverse_temperature=inverse_temperature)
            if it < iterations - 1:
                rr = e_step(o_mean, o_stdv, o_activations, votes)

        # pose: [N, OH, OW, O, PH x PW] via squeeze o_mean [N, OH, OW, 1, O, PH x PW]
        poses = tf.squeeze(o_mean, axis=-3)

        # activation: [N, OH, OW, O] via squeeze o_activationis [N, OH, OW, 1, O, 1]
        activations = tf.squeeze(o_activations, axis=[-3, -1])

    return poses, activations