import gradio as gr
from generate_message import generate_messages

def greet(input_query, overview, top_k, api_type, temperature):
    response, top1_example = generate_messages(input_query, overview, top_k, api_type, temperature)
    return response , top1_example

iface = gr.Interface(
    fn=greet,
    title="<h1 style='text-align:center; color: #c8c7d0;font-weight:700'> 국민은행 마케팅 카피라이터 AI</h1>",
    description = """<div style='text-align:center'><div style='display:inline-block;vertical-align:top'>
<a href='https://imgbb.com/'><img src='https://i.ibb.co/WWVQ3SN/kb-logo.webp' alt='kb-logo' style='width:95px; height:95px;margin-right:25px;border-radius:11px;'></a>
</div>
<div style="display:inline-block;margin-top: 14px;">
<p style='color: #ffbc00;font-family: -apple-system, BlinkMacSystemFont, sans-serif;font-weight:750;font-size:40px'>KB COPT</p>
</div></div>""",
    inputs=[gr.Textbox(label='한 줄 요약', info="생성하고자 하는 마케팅 메세지에 대해 한 줄로 입력해주세요."),
            gr.Textbox(label='세부 개요', info="생성하고자 하는 마케팅 메세지의 세부적인 개요를 입력해주세요."),
            gr.Radio(["No example", "Top 1", "Top 2", "Top 3"]),
            gr.Radio(["gpt-3.5-turbo", "gpt-4"], label="ChatGPT API", info="어떤 API를 활용하여 문구를 생성하시겠습니까?"),
            gr.Slider(0, 1,value=0,label='ChatGPT API Temperature', info="temperature가 클수록 메세지는 다양하게 생성됩니다.")],
    examples=[
        ['KB국민은행의 마케팅 메시지로, 고객들에게 행운의 상자 이벤트 참여를 유도하고 3천만원 추첨 경품을 제공하여 고객들의 참여를 유도하는 것이 목적입니다.', "- 기간: 2023년 8월 14일(월)부터 8월 18일(금)까지의 5일간 진행됩니다.\n- 응모방법: 고객들은 KB PAY 앱을 통해 행운의 상자를 열고 상품을 확인한 후 응모해야 합니다.\n- 리워드: 행운의 상자를 열고 응모한 일부 참여자들에게는 상품이 제공되며, 전체 응모자 중 1명이 추첨되어 3천만원의 현금이 제공됩니다.\n- 그 외 정보: 이벤트는 KB국민은행에서 주최하며, 당첨자 발표일은 2023년 8월 25일(금)이며, 경품은 발표일로부터 10영업일 이내에 지급됩니다. 이벤트에 대한 자세한 문의는 고객센터(☎1234-7777)로 문의하면 됩니다."],
        ],
    outputs=[gr.Textbox(label='생성된 마케팅 메세지'), gr.Textbox(label='가장 비슷한 샘플 메세지', info="코사인 유사도 기반 가장 유사한 메세지입니다.")],
    theme='finlaymacklon/smooth_slate'
    )

iface.launch()

