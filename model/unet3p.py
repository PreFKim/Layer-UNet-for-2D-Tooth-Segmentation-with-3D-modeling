
import keras

from .utils import conv

def unet3(n_levels,DSV=True,CGM=True, initial_features=32, n_blocks=2, kernel_size=3, pooling_size=2, IMAGE_HEIGHT=400, IMAGE_WIDTH=400, in_channels=1, out_channels=1):
    inputs = keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, in_channels))
    x = inputs

    #인코더부분
    skips = {}
    output = []
    for level in range(n_levels):
        if level != 0 :
            x = keras.layers.MaxPool2D(pooling_size,pooling_size)(x)
        for _ in range(n_blocks):
            x = conv(x,initial_features * 2 ** level,3,1,'same')
        skips[level] = x

        if level == n_levels-1:
            output.append(x)

    # CGM
    # if CGM:
    #     cgm = keras.layers.Dropout(0.5)(output[0])
    #     cgm = keras.layers.Conv2D(2, 1,1, padding='same')(cgm)
    #     cgm = keras.layers.GlobalMaxPooling2D()(cgm)
    #     cgm = keras.activations.sigmoid(cgm)

    #디코더 부분
    for level in reversed(range(n_levels-1)):
        list_concat = []

        #위에 레벨+ 동일레벨
        for i in range(level+1):
            x = skips[i]
            x = keras.layers.MaxPool2D(pooling_size**(level-i),pooling_size**(level-i))(x)
            x = conv(x,initial_features,3,1,'same')
            list_concat.append(x)

        #아래 레벨
        for i in range(level+1,n_levels):
            x = output[n_levels-i-1]
            x = keras.layers.UpSampling2D(pooling_size**(i-level), interpolation='bilinear')(x)
            x = conv(x,initial_features,3,1,'same')
            list_concat.append(x)

        #concat 부분
        x = keras.layers.concatenate(list_concat)
        for i in range(1): #논문에서는 1회만 진행
            x = conv(x,initial_features * n_levels,3,1,'same',is_bn = True)
        output.append(x)

    # 출력부분
    result = []
    if DSV:
        for i in range(n_levels):
            output[i] = conv(output[i],out_channels,3,1,'same',False,False)
            output[i] = keras.layers.UpSampling2D(2**(n_levels-i-1), interpolation='bilinear')(output[i])
            result.append(output[i])
    else:
        result.append(conv(output[n_levels-1],out_channels,3,1,'same',False,False))

    for i in range(len(result)):
        if out_channels == 1:
            result[i] = keras.activations.sigmoid(result[i])
        else:
            result[i] = keras.activations.softmax(result[i])

    #이름 설정
    output_name=f'UNET3-L{n_levels}-F{initial_features}'
    if DSV:
        output_name+='-DSV'

    # if CGM:
    #     output_name+='-CGM'
    #     result.append(cgm)

    return keras.Model(inputs=[inputs], outputs=result, name=output_name)

