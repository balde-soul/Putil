model_base.py:
    gen_pro:
        provide method to generate the pro of yolo2
    append_yolo2_loss:
        provide method to generate the loss of yolo2
    _PlaceGT:
        provide method to generate the placeholder
    _place_process:
        provide method to process the placeholder in order to build the loss easier
    _split_pro_ac:
        provide method to process the pro in order to build the loss easier
    _pro_result_reader:
        generate the special pro in order to build the loss easier
    _calc_iou:
        calc the iou label for using in building loss
    _calc_loss:
        build the yolo2 loss

    general usage:
        # the output from the base model
        feature_feed = tf.placeholder(dtype=tf.float32, shape=[10, 10, 10, 100], name='other_net_feature')
        f = np.zeros([10, 10, 10, 100], np.float32)
        # for feed
        # mask represent the cell which has obj(obj: 1; nobj: 0)
        p_mask = np.zeros([10, 10, 10, 1], np.float32)
        # mask represent the cell which does not have obj(obj: 0; nobj: 1)
        n_mask = np.ones([10, 10, 10, 1], np.float32)
        # provide the class data for each anchor(illegal anchor and nobj cell get random class data which would not
        # take part in loss calc
        cl = np.zeros([10, 10, 10, 4], np.int64)
        # provide the y data for each anchor(illegal anchor and nobj cell get random y data which would not
        # take part in loss calc
        y = np.zeros([10, 10, 10, 4], np.float32)
        # provide the x data for each anchor(illegal anchor and nobj cell get random x data which would not
        # take part in loss calc
        x = np.zeros([10, 10, 10, 4], np.float32)
        # provide the h data for each anchor(illegal anchor and nobj cell get random h data which would not
        # take part in loss calc
        h = np.zeros([10, 10, 10, 4], np.float32)
        # provide the w data for each anchor(illegal anchor and nobj cell get random w data which would not
        # take part in loss calc
        w = np.zeros([10, 10, 10, 4], np.float32)
        # mask represent the illegal and legal anchor(illegal: 0; legal: 1)
        anchor_mask = np.zeros([10, 10, 10, 4], np.float32)
        # 3: class_amount, 4: anchor_amount
        yolo_feature = gen_pro(feature_feed, 3, 4)
        # 3: class_amount, [10, 5, 3, 4]: prior anchor height, [2, 3, 4, 8]: prior anchor width, 32: base model pool scalar
        loss, place = append_yolo2_loss(yolo_feature, 3, [10, 5, 3, 4], [2, 3, 4, 8], 32)
        tf.summary.FileWriter('../test/yolo/model_base-', tf.Session().graph).close()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        print(sess.run([loss], feed_dict={
            feature_feed: f,
            place['class']: cl,
            place['y']: y,
            place['x']: x,
            place['h']: h,
            place['w']: w,
            place['p_mask']: p_mask,
            place['n_mask']: n_mask,
            place['anchor_mask']: anchor_mask
        }))
    implement method:
        yolo2 feature:
            base on the base model, add a conv layer which has kernel (1, 1) and stride (1, 1),
            generate pro channel: (4 + 1 + class_amount) * anchor_amount.
            base on the base model pool scalar, every spacial element is responsible for predict the obj which center
            locates in the element pool field.
            iterator cluster number n, and cluster the height and width on the ground truth height and width,
            base on every n calc the mIoU with the ground truth(make the center locating at the same place of every
            ground truth, and calc the mIoU), get the base n and the [height , width], this is the prior anchor.
            how to calculation loss:
                anchor loss[y, x, h, w]:
                    prior anchor which surpasses the edge of the image and cell which do not hold any obj
                    do not take part in the anchor loss calculation, this is why we need anchor_mask.
                    loss is base on square error.
                    calc the sum base on the anchor_obj_mask and then divide by anchor_obj_amount
                precision loss:
                    prior anchor which surpasses the edge of the image do not take part in the precision loss
                    calculation.
                    what is the precision label:
                        this label is calculation by the pro y, x, h, w and gt_y, gt_x, gt_h, gt_w, calculation the IoU
                        to represent the precision, cell which does not hold any obj place the 0 label.(see _calc_iou
                        function to get more implement detail, important: what we feed into the network is nt the real
                        parameter in the image, we should make some process, see in the _calc_iou function).
                    calc the sum base on the obj_mask and then divide by anchor_obj_amount
                class loss:
                    prior anchor which surpasses the edge of the image and cell which do not hold any obj do not
                    take part in the class loss calculation.
                    important: what we feed into the network is not a one-hot data, we process it into one-hot(see
                    _place_process for more detail) this cause of mask using difficult, we should reshape the pro
                    and the mask and multiply them.
                    calc the sum base on the anchor_obj_mask and the divide by anchor_obj_amount
                three losses has a lambda parameter, we multiply it and then make the sum , get the final loss

                this loss is being verified and has some difference with the yolo2 paper




