import keras

from .utils import conv


def layerUNET(n_levels, DSV=True ,initial_features=32, n_blocks=2, kernel_size=3, pooling_size=2, IMAGE_HEIGHT=400, IMAGE_WIDTH=400, in_channels=3, out_channels=1):
    inputs = keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, in_channels))
    x = inputs

    skips = []
    for _ in range(n_levels):
      skips.append(list())

    #인코더부분
    for level in range(n_levels):
      if level != 0 :
        x = keras.layers.MaxPool2D(pooling_size)(x)
      for _ in range(n_blocks):
        x = conv(x,initial_features * 2 ** level,kernel_size,1,'same')
      skips[level].append(x)

    #스킵 생성 부분
    for i in range(1,n_levels):
      for level in range(n_levels-i):
        list_concat = []

        #같은 레벨
        x = conv(skips[level][0],initial_features,kernel_size,1,'same')
        list_concat.append(x)

        #위에레벨
        for row in range(level):
          x = keras.layers.MaxPool2D(pooling_size**(level-row))(skips[row][0])
          x = conv(x,initial_features,3,1,'same')
          list_concat.append(x)
        
        #아래 레벨
        for j in range(i):
          x = keras.layers.UpSampling2D(pooling_size**(j+1), interpolation='bilinear')(skips[level+j+1][i-j-1])
          x = conv(x,initial_features,3,1,'same')
          list_concat.append(x)
        
        if (i>1):
          tmp = []
          for j in range(1,i):
            tmp.append(skips[level][j])
          if (len(tmp)>1):
            x = keras.layers.concatenate(tmp)
          else :
            x = tmp[0]
          x = conv(x,initial_features,3,1,'same',is_relu =True)
          list_concat.append(x)

        #Concatenate 부분
        x = keras.layers.concatenate(list_concat)
        for _ in range(1): #1회가 평균적으로 성능은 더 좋음
          x = conv(x,initial_features * len(list_concat),3,1,'same',is_bn= True)
        skips[level].append(x)
    
    # 출력부분
    result = []
    
    if DSV:
      for i in range(1,n_levels):
        x = keras.layers.Conv2D(out_channels, kernel_size=1, padding='same',kernel_initializer='he_normal')(skips[0][i])
        result.append(x)

      for i in range(1,n_levels):
        x = keras.layers.Conv2D(out_channels, kernel_size=3, padding='same',kernel_initializer='he_normal')(skips[i][-1])
        x = keras.layers.UpSampling2D(pooling_size**(i), interpolation='bilinear')(x)
        result.append(x)
    else:
      result.append(keras.layers.Conv2D(out_channels, kernel_size=1, padding='same',kernel_initializer='he_normal')(skips[0][-1]))
    
    for i in range(len(result)):
      if out_channels == 1:
        result[i] = keras.activations.sigmoid(result[i])
      else:
        result[i] = keras.activations.softmax(result[i])

    output_name=f'LayerUNET-L{n_levels}-F{initial_features}'

    if DSV:
      output_name+=f'-DSV'

    return keras.Model(inputs=[inputs], outputs=result, name=output_name)