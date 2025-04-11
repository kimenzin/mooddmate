import streamlit as st
import pandas as pd
from model_utils import load_sentiment_model

# 모델 로드
classifier = load_sentiment_model()

# 헤더
st.title("문장 감성 분석기")
st.write("이 앱은 입력한 문장의 감성을 분석하여, 긍정, 부정, 중립을 판단합니다.")

# 사용자 입력 받기
user_input = st.text_input("감성 분석할 문장을 입력하세요:")

# 분석 버튼 클릭 시
if st.button("분석하기"):
    if user_input:
        # 모델을 사용하여 텍스트 분석
        result = classifier(user_input)

        # 예측 결과와 확신도 출력
        label = result[0]['label']
        score = result[0]['score']

        # 감성 분석 결과 출력
        if label == "POSITIVE":
            st.write("🙂 긍정적인 문장입니다!")
        elif label == "NEGATIVE":
            st.write("😞 부정적인 문장입니다.")
        else:
            st.write("😐 중립적인 문장입니다.")
        
        st.write(f"확신도: {score:.2%}")
    else:
        st.warning("먼저 문장을 입력해주세요!")

# CSV 데이터 분석 예시
st.write("### 예시 데이터 감성 분석")
try:
    data = pd.read_csv("sentiment_data.csv")  # 파일은 레포지토리 루트에 있어야 함
    results = []
    for text in data['text']:
        result = classifier(text)
        label = result[0]['label']
        score = result[0]['score']
        results.append({'text': text, 'predicted_label': label, 'score': score})

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)
except FileNotFoundError:
    st.error("⚠️ 'sentiment_data.csv' 파일을 찾을 수 없습니다. 레포지토리에 업로드되어 있어야 합니다.")
