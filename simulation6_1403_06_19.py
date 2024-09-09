
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

points = []
start_x = 65
start_y = [0 , -5 , -10.0 ]
for i in range(15) :
  start_x +=  15

  for j in range(3) :
    points.append([start_x , start_y[j] , 0 ] )

points.append([330 , -2.5 , 0 ])
points.append([330 , -7.5 , 0 ])
points.append([330 , -12.5 , 0 ])
total_points = np.array(points)
plt.scatter(total_points[ : ,  0 ] , total_points[ : , 1 ] , label = "target" )
plt.legend()
plt.title("Blade Shape - Picture 1.7 page 7 document volume 2 ")
plt.xlabel(" X (in) ")
plt.ylabel(" Y (in) ")
plt.grid()
plt.show()


dict_28_24_blade_1 = {"r/R"           :[0.2001 , 0.2499 ,0.2998 , 0.3498 , 0.3913 , 0.4496 , 0.4997 , 0.5387 , 0.5996 , 0.6498 , 0.6998 , 0.7500 , 0.8001 , 0.8497 , 0.9102 , 0.9734 ] ,
              "Delta XMean (in.)"     : [-0.0045 , -0.0011 , -0.0281 , -0.0177 , -0.0113 , -0.0186 , -0.0133 , -0.0232 , -0.0338 , -0.0307 , -0.0633 , -0.0773 , -0.0811 , -0.1920 , -0.2623 , -0.2629 ] ,
              "Delta XStd (in.)"      : [0.0010 , 0.0009 , 0.0011 , 0.0011 , 0.0016 , 0.0019 , 0.0022 , 0.0020 , 0.0019 , 0.0023 , 0.0021 , 0.0021 , 0.0021 , 0.0022 , 0.0023 , 0.0026 ] ,
              "Delta YMean (in.)"     : [-0.0132 , -0.0225 , -0.0483 , -0.0939 , -0.1256 , -0.1809 , -0.2209 , -0.2510 , -0.3087 , -0.3422 , -0.3874 , -0.4184 , -0.5123 , -0.5296 , -0.5487 , -0.5354 ] ,
              "Delta YStd (in.)"      : [0.0007 , 0.0013 , 0.0028 , 0.0046 , 0.0066 , 0.0088 , 0.0111 , 0.0126 , 0.0150 , 0.0176 , 0.0201 , 0.0229 , 0.0258 , 0.0284 , 0.0313 , 0.0350 ]  ,
              "Delta ZMean (in.)"     : [-0.0050 , -0.0010 , -0.0251  ,-0.0442 , -0.0622 , -0.1306 , -0.2121 , -0.3077 , -0.4860 , -0.7122 , -0.9213 , -1.1503 , -1.3874 , -1.7166 , -2.1240 , -2.5162 ] ,
              "Delta ZStd (in.)"      : [0.0023 , 0.0046 , 0.0089 , 0.0145 , 0.0211 , 0.0267 , 0.0347 , 0.0404 , 0.0480 , 0.0540 , 0.0607 , 0.0666 , 0.0718 , 0.0784 , 0.0857 , 0.0943 ] ,
              "Delta TwistMean (deg)" : [-0.0685 , 0.2730 , 0.0049  , 0.0060 , 0.0002 , -0.1528 , -0.1632 , -0.3044 , -0.2646 , -0.4566 , -0.4899 , -0.3256 , 0.0164 , -0.3486 , -0.6888 , -0.6556 ] ,
              "Delta TwistStd (deg)"  : [0.0233 , 0.0392 , 0.0515 , 0.0431 , 0.0476 , 0.0461 , 0.0479 , 0.0528 , 0.0516 , 0.0503 , 0.0581 , 0.0443 , 0.0504 , 0.0462  , 0.0444 , 0.0510]
}


table_28_24_blade_1 = pd.DataFrame(dict_28_24_blade_1)




total_location = list(range( 10 , 15))
mizan_crack    = []
base_crack = 1.2
max_crack  = 2


while base_crack <= max_crack :
    mizan_crack.append(base_crack)
    base_crack += 0.1



params_health = []
params_fault  = []
df_new_like   = []
for oo in range(1) :



  for locate in total_location:
      for crack in mizan_crack:

          related_x = 1
          related_y = 1
          related_z = 1

          for iteration in range(100):
              alpha = 0
              term  = 0
              new_data_simulated_from_table_28_24 = []
              for i in range(table_28_24_blade_1.shape[0]):
                  x_mean = table_28_24_blade_1.iloc[i, 1]
                  x_std  = table_28_24_blade_1.iloc[i, 2]
                  y_mean = table_28_24_blade_1.iloc[i, 3]
                  y_std  = table_28_24_blade_1.iloc[i, 4]
                  z_mean = table_28_24_blade_1.iloc[i, 5]
                  z_std  = table_28_24_blade_1.iloc[i, 6]



                  if i > locate :
                      alpha = 0.004
                      term  = alpha * abs( i - locate )

                  for element in total_points[i * 3: (i + 1) * 3, :]:
                      x_sample = element[0]
                      y_sample = element[1]
                      z_sample = element[2]

                      rng = np.random.default_rng()

                      x_added  = rng.normal(x_mean, x_std, size=1)
                      x_added  = x_added.tolist()[0]
                      x_sample += related_x * x_added

                      y_added  = rng.normal(y_mean, y_std, size=1)
                      y_added  = y_added.tolist()[0]
                      y_sample += related_y * y_added

                      z_added  = rng.normal(z_mean, z_std, size=1)
                      z_base   = z_sample

                      z_added_term  = z_added.tolist()[0] + term
                      z_added       = z_added.tolist()[0] 

                      z_sample_new_term = z_base + related_z * z_added_term
                      z_sample_new      = z_base + related_z * z_added
                      
                      new_data_simulated_from_table_28_24.append([ x_sample, y_sample, z_sample_new ])
                      df_new_like.append([ x_sample, y_sample, z_sample_new_term ])

              new_data_simulated_from_table_28_24 = np.array(new_data_simulated_from_table_28_24)
              new_data_simulated_from_table_28_24_fault = np.array(df_new_like)


              U1, V1 = np.meshgrid(new_data_simulated_from_table_28_24[:, 0], new_data_simulated_from_table_28_24[:, 1])
              U, V = np.meshgrid(total_points[:, 0], total_points[:, 1])

              #fig = plt.figure(figsize=(50, 15))

              # ax01 = fig.add_subplot(1 , 6 , 1 , projection='3d' )
              # ax02 = fig.add_subplot(1 , 6 , 2 , projection='3d' )
              # ax03 = fig.add_subplot(1 , 6 , 3 , projection='3d' )
              # ax1 = fig.add_subplot(1 , 6 , 4 , projection='3d' )
              # ax01.plot_surface(U1 , V1 , new_data_simulated_from_table_28_24[ : ,  2 ].reshape(1 , -1 )  , alpha = 0.1  )
              # ax01.view_init(elev=5., azim=-85)

              # ax02.plot_surface(U1 , V1 , new_data_simulated_from_table_28_24[ : ,  2 ].reshape(1 , -1 )  , alpha = 0.1  )
              # ax02.view_init(elev=20, azim=-50)

              # ax03.plot_surface(U1 , V1 , new_data_simulated_from_table_28_24[ : ,  2 ].reshape(1 , -1 )  , alpha = 0.1  )
              # ax03.view_init(elev=25, azim=-45)

              # ax1.scatter3D(new_data_simulated_from_table_28_24[ : ,  0 ], new_data_simulated_from_table_28_24[ : , 1 ] , new_data_simulated_from_table_28_24[ : , 2  ] , color = "red" )
              # ax1.scatter3D(total_points[ : ,  0 ], total_points[ : , 1 ] , total_points[ : , 2  ] , color = "aqua" )
              df_health = pd.DataFrame()
              df_health['X_'] = new_data_simulated_from_table_28_24[:, 0]
              df_health['Y_'] = new_data_simulated_from_table_28_24[:, 1]
              df_health['Z_'] = new_data_simulated_from_table_28_24[:, 2]

              poly = PolynomialFeatures(degree=2)
              x_poly = poly.fit_transform(df_health[['X_', 'Y_']])
              scaler = StandardScaler()
              df_stander = pd.DataFrame(scaler.fit_transform(x_poly), columns=['0', 'X', 'Y', 'x1^2', 'x1x2', 'x2^2', ])

              df_stander['Z_'] = df_health['Z_']
              x_train, x_test, z_train, z_test = train_test_split(df_stander[['0', 'X', 'Y', 'x1^2', 'x1x2', 'x2^2']],
                                                                  df_stander["Z_"])
              model = LinearRegression()
              model.fit(x_train, z_train)

              z_predict = model.predict(x_test)
              print("Mse is : ", mean_squared_error(z_predict, z_test))

              intercept_0 = model.intercept_
              coef1 = model.coef_[0]
              coef2 = model.coef_[1]
              coef3 = model.coef_[2]
              coef4 = model.coef_[3]
              coef5 = model.coef_[4]
              coef6 = model.coef_[5]

              total_new_generate = []
              total_error = []
              for i in range(len(df_health['X_'])):
                  new_value = intercept_0 + coef1 * df_stander.iloc[i, 0] + coef2 * df_stander.iloc[i, 1] + coef3 * df_stander.iloc[i, 2] + coef4 * df_stander.iloc[i, 3] + coef5 * df_stander.iloc[i, 4]
                  + coef6 * df_stander.iloc[i, 5]
                  total_new_generate.append(new_value)
                  error = df_stander.iloc[i, -1] - new_value
                  total_error.append(error)

              # ax5 = fig.add_subplot(1 , 6 , 5 , projection='3d' )
              # U5_meshgrid , V5_meshgrid = np.meshgrid(df['X_'], df[ 'Y_' ])
              # ax5.plot_surface( U5_meshgrid , V5_meshgrid , np.array(total_new_generate).reshape(1 , -1 ) )

              # ax6 = fig.add_subplot(1 , 6 , 6 , projection='3d' )
              # ax6.scatter3D( new_data_simulated_from_table_28_24[ : ,  0 ], new_data_simulated_from_table_28_24[ : , 1 ] , new_data_simulated_from_table_28_24[ : , 2  ] , color = "red" )
              # ax6.scatter3D( new_data_simulated_from_table_28_24[ : ,  0 ], new_data_simulated_from_table_28_24[ : , 1 ] ,  np.array(total_new_generate).reshape(1 , -1 ) , color = "b" )

              # fig2 = plt.figure(figsize = (15 , 5 ))
              # ax9 = fig2.add_subplot(111 )
              # ax9.scatter(range(len(df)) , total_error   , label='error')
              # ax9.legend()
              # ax9.grid()

              params_health.append(
                  [intercept_0, coef1, coef2, coef3, coef4, coef5, coef6, mean_squared_error(z_predict, z_test),
                   locate  ])
              total_change = [1, 1]
              related_x *= total_change[np.random.randint(0, 2)]
              related_y *= total_change[np.random.randint(0, 2)]
              related_z *= total_change[np.random.randint(0, 2)]
              #print(f"location : {locate} - {crack}")
              # plt.show()











              print(" ----------------------------------- calculate fault ------------------------------------")



              df_fault = pd.DataFrame()
              df_fault['X_'] = new_data_simulated_from_table_28_24_fault[:, 0]
              df_fault['Y_'] = new_data_simulated_from_table_28_24_fault[:, 1]
              df_fault['Z_'] = new_data_simulated_from_table_28_24_fault[:, 2]

              poly_fault = PolynomialFeatures(degree=2)
              x_poly_fault = poly_fault.fit_transform(df_fault[['X_', 'Y_']])
              scaler_fault = StandardScaler()
              df_stander_fault = pd.DataFrame(scaler_fault.fit_transform(x_poly_fault), columns=['0', 'X', 'Y', 'x1^2', 'x1x2', 'x2^2', ])

              df_stander_fault['Z_'] = df_fault['Z_']
              x_train_fault , x_test_fault , z_train_fault , z_test_fault = train_test_split(df_stander_fault[['0', 'X', 'Y', 'x1^2', 'x1x2', 'x2^2']], df_stander_fault["Z_"])
              model_fault = LinearRegression()
              model_fault.fit(x_train_fault, z_train_fault)

              #z_predict_fault = model.predict(x_test_fault)
              #print("Mse is calculate fault : ", mean_squared_error(z_predict, z_test_fault ))

              intercept_0 = model_fault.intercept_
              coef1 = model_fault.coef_[0]
              coef2 = model_fault.coef_[1]
              coef3 = model_fault.coef_[2]
              coef4 = model_fault.coef_[3]
              coef5 = model_fault.coef_[4]
              coef6 = model_fault.coef_[5]

              total_new_generate_fault = []
              total_error_fault = []
              for i in range(len(df_fault['X_'])):
                  new_value = intercept_0 + coef1 * df_stander_fault.iloc[i, 0] + coef2 * df_stander_fault.iloc[i, 1] + coef3 *  df_stander_fault.iloc[i, 2] + coef4 * df_stander_fault.iloc[i, 3] + coef5 * df_stander_fault.iloc[i, 4]  + coef6 * df_stander_fault.iloc[i, 5]
                  total_new_generate_fault.append(new_value)
                  error_Fault = df_stander_fault.iloc[i, -1] - new_value
                  total_error_fault.append(error_Fault)

              # ax5 = fig.add_subplot(1 , 6 , 5 , projection='3d' )
              # U5_meshgrid , V5_meshgrid = np.meshgrid(df['X_'], df[ 'Y_' ])
              # ax5.plot_surface( U5_meshgrid , V5_meshgrid , np.array(total_new_generate).reshape(1 , -1 ) )

              # ax6 = fig.add_subplot(1 , 6 , 6 , projection='3d' )
              # ax6.scatter3D( new_data_simulated_from_table_28_24[ : ,  0 ], new_data_simulated_from_table_28_24[ : , 1 ] , new_data_simulated_from_table_28_24[ : , 2  ] , color = "red" )
              # ax6.scatter3D( new_data_simulated_from_table_28_24[ : ,  0 ], new_data_simulated_from_table_28_24[ : , 1 ] ,  np.array(total_new_generate).reshape(1 , -1 ) , color = "b" )

              # fig2 = plt.figure(figsize = (15 , 5 ))
              # ax9 = fig2.add_subplot(111 )
              # ax9.scatter(range(len(df)) , total_error   , label='error')
              # ax9.legend()
              # ax9.grid()

              params_fault.append(
                  [intercept_0, coef1, coef2, coef3, coef4, coef5, coef6, mean_squared_error(z_predict, z_test),
                   locate  ])
              total_change = [1, 1]
              related_x *= total_change[np.random.randint(0, 2)]
              related_y *= total_change[np.random.randint(0, 2)]
              related_z *= total_change[np.random.randint(0, 2)]
              #print(f"location : {locate} - {crack}")
              # plt.show()
              print("oo" , oo , " , locate : " , locate , " , crack : " , crack , " , iter : " , iteration ) 






params_health = np.array(params_health)
df_params = pd.DataFrame({'0':params_health[: , 0 ] , '1':params_health[ : , 1 ] , '2':params_health[ : , 2 ] , '3':params_health[ : , 3 ] ,  '4':params_health[ : , 4 ] , '5':params_health[ : , 5 ] , '6':params_health[ : , 6 ] , 'error' : params_health[ : , 7 ]  ,
                          'locate' : params_health[: , 8 ].astype(int) }   )

df_params['class'] = 1


params_fault = np.array(params_fault)
df_params_fault = pd.DataFrame({'0':params_fault[: , 0 ] , '1':params_fault[ : , 1 ] , '2':params_fault[ : , 2 ] , '3':params_fault[ : , 3 ] ,  '4':params_fault[ : , 4 ] , '5':params_fault[ : , 5 ] , '6':params_fault[ : , 6 ] , 'error' : params_fault[ : , 7 ]  ,
                          'locate' : params_fault[: , 8 ].astype(int) }   )


df_params_fault['class'] = 0 

concat_2_df_health_fault = pd.concat([df_params , df_params_fault ] , axis = 0 )






#df_params['Shedat*locate']  =  df_params['values'] * df_params['locate']







df_corr = concat_2_df_health_fault.corr()



import seaborn as sns
plt.figure(figsize= (10 , 10 ) )
ax = sns.heatmap(df_corr , annot=True )
print(df_corr)
plt.show()

