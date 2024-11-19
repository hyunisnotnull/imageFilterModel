# 한글 폰트 설치하기
!apt install fonts-nanum -y

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정하기
fontpath = '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'

# 이미지 크롤링에 필요한 bing image downloader 설치하기
!git clone https://github.com/ndb796/bing_image_downloader

# 이미지 크롤링을 활용한 학습 이미지 수집
# 수집한 이미지를 저장하기 위한 폴더 생성, 필요한 함수 정의
import os
import shutil
from bing_image_downloader.bing_image_downloader import downloader

directory_list = [
    './custom_dataset/train/',
    './custom_dataset/test/',
]

# 초기 디렉토리 만들기
for directory in directory_list:
    if not os.path.isdir(directory):
        os.makedirs(directory)

# 수집한 이미지를 학습 데이터와 평가 데이터로 구분하는 함수
def dataset_split(query, train_cnt):
    # 학습 및 평가 데이터셋 디렉토리 만들기
    for directory in directory_list:
        if not os.path.isdir(directory + '/' + query):
            os.makedirs(directory + '/' + query)
    # 학습 및 평가 데이터셋 준비하기
    cnt = 0
    for file_name in os.listdir(query):
        if cnt < train_cnt:
            print(f'[Train Dataset] {file_name}')
            shutil.move(query + '/' + file_name, './custom_dataset/train/' + query + '/' + file_name)
        else:
            print(f'[Test Dataset] {file_name}')
            shutil.move(query + '/' + file_name, './custom_dataset/test/' + query + '/' + file_name)
        cnt += 1
    shutil.rmtree(query)

# 원하는 이미지를 크롤링하고 데이터셋을 구축하는 부분
queries = ['술', '담배', '약', '현금', '총', '통장', '신분증']
for query in queries:
    try:
        downloader.download(query, limit=40, output_dir='./', adult_filter_off=True, force_replace=False, timeout=60)
        dataset_split(query, 30)
    except Exception as e:
        print(f"이미지 크롤링 실패: {query} - {e}")

# 필요한 라이브러리 임포트
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import time
import os

# GPU 사용 여부 확인
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 데이터셋을 불러올 때 사용할 객체 정의
transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),  # 224x224 크기로 데이터를 맞춤
    transforms.RandomHorizontalFlip(),  # 데이터 증진 : 이미지의 좌우반전을 맞게 해줌
    transforms.ToTensor(),  # 이미지 데이터를 텐서 객체로 변환
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 정규화
])

transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 데이터셋 폴더 경로
data_dir = './custom_dataset'
train_datasets = datasets.ImageFolder(os.path.join(data_dir, 'train'), transforms_train)
test_datasets = datasets.ImageFolder(os.path.join(data_dir, 'test'), transforms_test)

# DataLoader 정의 (배치 크기 8, shuffle=True)
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=8, shuffle=True, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=8, shuffle=True, num_workers=4)

# 클래스 이름 출력
class_names = train_datasets.classes
print('클래스:', class_names)

# CNN 모델 설정 (ResNet34)
model = models.resnet34(pretrained=True)
num_features = model.fc.in_features

# 전이 학습 : 모델의 출력 뉴런 수를 7개로 교체
model.fc = nn.Linear(num_features, len(class_names))  # class_names 수에 맞게 출력 노드 수 조정
model = model.to(device)  # 모델을 GPU/CPU로 이동

# 손실 함수 및 최적화 방법 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 학습률 스케줄러 설정 (매 7번째 에폭마다 학습률을 10배 감소)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 학습 파라미터
num_epochs = 50
model.train()  # 학습 모드

# 학습 시작
start_time = time.time()

for epoch in range(num_epochs):
    running_loss = 0.0
    running_corrects = 0

    # 배치 단위로 학습 진행
    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 모델에 입력하고 출력 계산
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # 역전파로 기울기 계산 및 학습
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    scheduler.step()  # 학습률 스케줄러 호출

    epoch_loss = running_loss / len(train_datasets)
    epoch_acc = running_corrects / len(train_datasets) * 100.

    # 학습 결과 출력
    print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}% Time: {time.time() - start_time:.4f}s')

# 모델 저장
torch.save(model.state_dict(), '/content/model.pth')

# 모델 파일 다운로드
from google.colab import files
files.download('/content/model.pth')

# 특성 추출 및 유사도 계산을 위한 코드 추가
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# 업로드된 이미지를 로드하는 함수
def load_image(image_path):
    img = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor.to(device)

# 특징 벡터 추출 함수
def get_features(image_tensor):
    with torch.no_grad():
        features = model(image_tensor)
    return features.cpu().numpy().reshape(1, -1)

# 카테고리별 대표 이미지 경로 (미리 준비된 이미지)
category_embeddings = {
    '술': '/content/custom_dataset/test/술/image_1.jpg',  # 술 카테고리의 이미지 경로
    '담배': '/content/custom_dataset/test/담배/image_11.jpg',  # 담배 카테고리의 이미지 경로
    '약': '/content/custom_dataset/test/약/image_3.jpg',  # 약 카테고리의 이미지 경로
    '현금': '/content/custom_dataset/test/현금/image_7.jpg',  # 현금 카테고리의 이미지 경로
    '총': '/content/custom_dataset/test/총/image_2.jpg',  # 총 카테고리의 이미지 경로
    '통장': '/content/custom_dataset/test/통장/image_4.jpg',  # 통장 카테고리의 이미지 경로
    '신분증': '/content/custom_dataset/test/신분증/image_5.jpg'  # 신분증 카테고리의 이미지 경로
}

# 카테고리별 특징 벡터 저장
category_features = {}

def extract_category_features():
    for category, image_path in category_embeddings.items():
        image_tensor = load_image(image_path)
        category_features[category] = get_features(image_tensor)

# 카테고리별 특징 벡터 추출
extract_category_features()

# 업로드된 이미지 처리 및 유사도 계산 함수
def process_uploaded_image(image_path, threshold=0.95):
    # 이미지를 로드하고 예측된 카테고리 찾기
    uploaded_image = load_image(image_path)  # 이미지 로드
    with torch.no_grad():
        output = model(uploaded_image)
        _, preds = torch.max(output, 1)

    # 예측된 카테고리
    predicted_class = class_names[preds.item()]
    print(f"예측된 카테고리: {predicted_class}")

    # 업로드된 이미지의 특징 벡터 추출
    uploaded_embedding = get_features(uploaded_image)

    # 해당 카테고리의 대표 이미지 특징 벡터와 비교하여 유사도 계산
    category_embedding = category_features[predicted_class]

    similarity = cosine_similarity(uploaded_embedding, category_embedding)

    # 유사도가 임계값 이상이면 거부
    if similarity >= threshold:
        print(f"이미지가 '{predicted_class}'와 유사도가 높아 거부되었습니다.")
        return False  # 등록 거부
    else:
        print("이미지가 카테고리와 유사도가 낮아 통과되었습니다.")
        return True  # 등록 허용

# 테스트를 위한 예시
image_path = '/content/custom_dataset/test/신분증/image_5.jpg'  # 테스트할 이미지 경로
process_uploaded_image(image_path)
