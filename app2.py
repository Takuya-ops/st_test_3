from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

st.title("専門家になんでも聞いて！")
st.write(
    "この質問コーナーは、健康維持について、理学療法士と管理栄養士の専門家に質問できるサービスです。"
)

selected_item = st.radio(
    "2人の専門家の中から、質問したい専門家を選んでください。",
    ["理学療法士", "管理栄養士"],
)
st.divider()

st.write(f"選択された専門家:{selected_item}")

input_message = st.text_input(
    label="健康に関するお悩みやその他、質問を下記の欄に入力してください。"
)


system_message = {
    "理学療法士": "あなたは理学療法士です。お客様の健康維持のため、適切なアドバイスで、身体的なサポートを提案し、リハビリテーションに関する専門家です。"
    "専門用語を使用せず、日本語で中学生でも理解できるように分かりやすく説明してください。200文字以内で回答してください。",
    "管理栄養士": "あなたは管理栄養士です。お客様の健康維持のため、適切なアドバイスで、栄養バランスの取れた食事プランを提案し、健康的なライフスタイルに関する専門家です。"
    "専門用語を使用せず、日本語で中学生でも理解できるように分かりやすく説明してください。200文字以内で回答してください。",
}
prompt = ChatPromptTemplate.from_messages(
    [("system", system_message[selected_item]), ("human", "{input}")]
)

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)

answer_chain = LLMChain(llm=llm, prompt=prompt)

point_prompt = ChatPromptTemplate.from_template(
    """あなたは質問に対して、専門的な知識がある専門家です。
    以下の質問が「健康」「身体」「栄養」「運動」「リハビリ」「メンタル」等に関する内容なら「はい」、
    全く関係ない内容であれば、「いいえ」とだけ答えてください。
    質問:{question}
    """
)
point_chain = LLMChain(llm=llm, prompt=point_prompt)

if st.button("実行"):
    st.divider()

    if not input_message:
        st.error("質問を入力してください。")

    else:
        point_result = point_chain.invoke({"question": input_message})
        if "いいえ" in point_result["text"]:
            st.write(
                "この質問は健康や栄養に関する内容ではないです。質問内容を変更してください。"
            )
        else:
            result = answer_chain.invoke({"input": input_message})
            st.write("### 回答:")
            st.write(result["text"])
