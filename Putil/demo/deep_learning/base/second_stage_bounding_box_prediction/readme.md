# second_stage_bounding_box_prediction
* 核心内容
  * 通过一阶段的bbox检测然后增加二阶段的细致回归
* 解决方案
  * [dcn_feature_calibration](./dcn_feature_calibration.py)
    * 基于one stage prediction生成Deformable Convolution Network的offset，[使用中心原则]
      $$
      prediction=(x_c,y_c,w,h,\theta) \\
      conv^{(3,3)} \\
      offset^{(18)} \\
      offset=[
          \frac{w}{4},
      ] \\
      given any positive-definite symmetrical 2*2 matrix Z: \\
      Tr(Z^(1/2)) = sqrt(\lambda_1) + sqrt(\lambda_2) \\
      where \lambda_1 and \lambda_2 are the eigen values of Z \\
      meanwhile we have: \\
      Tr(Z) = \lambda_1 + \lambda_2 \\
      det(Z) = \lambda_1 * \lambda_2 \\
      combination with following formula: \\
      (sqrt(\lambda_1) + sqrt(\lambda_2))^2 = \lambda_1 + \lambda_2 + 2 * sqrt(\lambda_1 * \lambda_2) \\
      yield: \\
      Tr(Z^(1/2)) = sqrt(Tr(Z) + 2 * sqrt(det(Z))) \\
      for gwd loss the frustrating coupling part is: \\
      Tr((Σp^(1/2) * Σt * Σp^(1/2))^(1/2)) \\
      assuming Z = Σp^(1/2) * Σt * Σp^(1/2) then: \\
      Tr(Z) = Tr(Σp^(1/2) * Σt * Σp^(1/2)) \\
      = Tr(Σp^(1/2) * Σp^(1/2) * Σt) \\
      = Tr(Σp * Σt) \\
      det(Z) = det(Σp^(1/2) * Σt * Σp^(1/2)) \\
      = det(Σp^(1/2)) * det(Σt) * det(Σp^(1/2)) \\
      = det(Σp * Σt) \\
      and thus we can rewrite the coupling part as: \\
      Tr((Σp^(1/2) * Σt * Σp^(1/2))^(1/2)) \\
      = Tr{Z^(1/2)} = sqrt(Tr(Z) + 2 * sqrt(det(Z)) \\
      = sqrt(Tr(Σp * Σt) + 2 * sqrt(det(Σp * Σt))) \\
      $$