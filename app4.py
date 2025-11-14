from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.title("専門家になんでも聞いて！")
st.write(
    "この質問コーナーは、健康維持について、理学療法士と管理栄養士の専門家に質問できるサービスです。"
)

# 専門家の選択
selected_item = st.radio(
    "2人の専門家の中から、質問したい専門家を選んでください。",
    ["理学療法士", "管理栄養士"],
)
st.divider()

st.write(f"選択された専門家: {selected_item}")

# 質問入力
input_message = st.text_input(
    label="健康に関するお悩みやその他、質問を下記の欄に入力してください。"
)

# システムメッセージ定義
system_message = {
    "理学療法士": (
        "あなたは理学療法士です。お客様の健康維持のため、適切なアドバイスで、"
        "身体的なサポートを提案し、リハビリテーションに関する専門家です。"
        "専門用語を使用せず、日本語で中学生でも理解できるように分かりやすく説明してください。"
        "200文字以内で回答してください。"
    ),
    "管理栄養士": (
        "あなたは管理栄養士です。お客様の健康維持のため、適切なアドバイスで、"
        "栄養バランスの取れた食事プランを提案し、健康的なライフスタイルに関する専門家です。"
        "専門用語を使用せず、日本語で中学生でも理解できるように分かりやすく説明してください。"
        "200文字以内で回答してください。"
    ),
}

# LLM 初期化（LCEL 用）
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
parser = StrOutputParser()

# 本回答用プロンプト & チェーン（LCEL）
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message[selected_item]),
        ("human", "{input}"),
    ]
)
answer_chain = answer_prompt | llm | parser

# 判定用プロンプト & チェーン（LCEL）
point_prompt = ChatPromptTemplate.from_template(
    """あなたは質問に対して、専門的な知識がある専門家です。
以下の質問が「健康」「身体」「栄養」「運動」「リハビリ」「メンタル」等に関する内容なら「はい」、
全く関係ない内容であれば、「いいえ」とだけ答えてください。

質問: {question}
"""
)
point_chain = point_prompt | llm | parser

if st.button("実行"):
    st.divider()

    if not input_message:
        st.error("質問を入力してください。")
    else:
        # 質問が健康関連かどうか判定
        point_result = point_chain.invoke({"question": input_message})

        if "いいえ" in point_result:
            st.write(
                "この質問は健康や栄養に関する内容ではないです。質問内容を変更してください。"
            )
        else:
            # 回答生成
            result = answer_chain.invoke({"input": input_message})
            st.write("### 回答:")
            st.write(result)
