import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from keras import datasets, layers, models


# MNIST 데이터셋 다운로드하고 준비하기
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# 픽셀 값을 0~1 사이로 정규화합니다.
train_images, test_images = train_images / 255.0, test_images / 255.0

# 합성곱 층 만들기
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2))) # 수직, 수평 축소 비율을 지정. (2,2)라면 출력 영상 크기는 입력 영상 크기의 반으로 줄임.
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

# 마지막에 Dense 층 추가하기
model.add(layers.Flatten()) # 1차원 자료로 바꿔줌.
#다층 퍼셉트론 신경망에서 사용되는 레이어로 입력과 출력을 모두 연결해준다. 
# 예를 들어, 입력 뉴런이 4개, 출력 뉴런이 8개라고 할때 총 연결선은 4x8=32개가 된다. 
# 각 연결선은 가중치(weight)를 포함하고 있는데 연결강도를 의미한다. 
# 가중치가 높을 수록 해당 입력 뉴런이 출력 뉴런에 미치는 영향이 크고, 낮을수록 미치는 영향이 작다.
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# 모델 컴파일과 훈련하기
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

hist = model.fit(train_images, train_labels, epochs=5)

#그래프 그리기

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'],'y',label='train loss')
# loss_ax.plot(hist.history['val_loss'],'r',label='val loss')
acc_ax.plot(hist.history['accuracy'],'b',label='train acc')
#acc_ax.plot(hist.history['val_accuracy'],'g',label='val acc')

loss_ax.set_xlabel('epochs')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()

'''
print(hist.history['loss'])
print(hist.history['acc'])
print(hist.history['val_loss'])
print(hist.history['val_acc'])
'''
# 모델 평가
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)

