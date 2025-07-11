import os
import tempfile
from nicegui import ui, app, run
from starlette.datastructures import UploadFile
from summarizer_unified import PDFSummarizer
from nicegui.events import UploadEventArguments

uploaded_file = {}
summary_label = None
keywords_label = None
tags_label = None
summary_text = ""

def run_summarization(file_path, max_keywords, max_tags, max_words, out_lang):
    summarizer = PDFSummarizer(llm_config={
        "api_key": os.getenv("SSP_KEY"),
        "temperature": 0.1,
    })
    return summarizer.process_pdf(
        file_path,
        max_keywords=max_keywords,
        max_tags=max_tags,
        max_words=max_words,
        out_lang=out_lang
    )

def handle_upload(e: UploadEventArguments):
    uploaded_file['file'] = e
    ui.notify(f'✅ Uploaded: {e.name}', type='positive')

async def handle_summarize(max_words_input, max_keywords_input, max_tags_input, out_lang_select, status_label):
    global summary_label, keywords_label, tags_label, summary_text

    if 'file' not in uploaded_file or uploaded_file['file'] is None:
        ui.notify('Please upload a PDF file first.', type='warning')
        return

    status_label.text = '⏳ Processing...'
    temp_file_path = None

    try:
        uploaded_file['file'].content.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            temp_file_path = tmp_file.name
            tmp_file.write(uploaded_file['file'].content.read())

        result = await run.cpu_bound(
            run_summarization,
            temp_file_path,
            int(max_keywords_input.value),
            int(max_tags_input.value),
            int(max_words_input.value),
            out_lang_select.value
        )

        summary_text = result.get("summary", "No summary.")
        summary_label.text = summary_text
        keywords = result.get("keywords", [])[:int(max_keywords_input.value)]
        tags = result.get("tags", [])[:int(max_tags_input.value)]
        keywords_label.text = ", ".join(keywords)
        tags_label.text = ", ".join(tags)
        status_label.text = "✅ Done"

    except Exception as e:
        summary_label.text = f"❌ Error: {str(e)}"
        keywords_label.text = "Error"
        tags_label.text = "Error"
        status_label.text = '❌ Failed'
        ui.notify(f"Summarization error: {str(e)}", type='negative')

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@ui.page('/')
def main_page():
    global summary_label, keywords_label, tags_label

    ui.add_head_html('<style>body { zoom: 100%; }</style>')

    with ui.column().classes('items-center p-10 gap-8 text-xl max-w-7xl w-full mx-auto'):

        ui.label('📄 PDF Summarizer with LLMs - WP12 - Statistics Portugal').classes('text-3xl font-bold')

        with ui.card().classes('w-full p-6 shadow-lg'):
            ui.label('📤 Upload your PDF').classes('text-2xl mb-2')
            ui.upload(on_upload=handle_upload, auto_upload=True)\
              .props('accept=".pdf" max-files=1 color=primary').classes('w-full')

        with ui.card().classes('w-full p-6 shadow-lg'):
            ui.label('⚙️ Parameters').classes('text-2xl mb-4')
            with ui.grid(columns=2).classes('gap-4 w-full'):
                max_words_input = ui.number('Max Words', value=200, min=50, max=1000).classes('w-full')
                max_keywords_input = ui.number('Max Keywords', value=6, min=1, max=20).classes('w-full')
                max_tags_input = ui.number('Max Tags', value=5, min=1, max=15).classes('w-full')
                out_lang_select = ui.select(['pt-pt', 'en'], value='pt-pt', label='Output Language').classes('w-full')

        status_label = ui.label('Ready.').classes('text-lg text-gray-600')

        ui.button('🔍 Summarize', on_click=lambda: handle_summarize(
            max_words_input, max_keywords_input, max_tags_input, out_lang_select, status_label
        )).classes('w-full bg-blue-600 text-white text-lg py-2 rounded hover:bg-blue-700')

        ui.separator().classes('w-full my-4')

        with ui.card().classes('w-full p-6 bg-gray-50'):
            ui.label('📄 Summary').classes('text-xl font-semibold')
            summary_label = ui.label('Summary will appear here.').classes('w-full whitespace-pre-wrap')

        with ui.card().classes('w-full p-6 bg-gray-50'):
            ui.label('🔑 Keywords').classes('text-xl font-semibold')
            keywords_label = ui.label('Keywords will appear here.').classes('w-full')

        with ui.card().classes('w-full p-6 bg-gray-50'):
            ui.label('🏷️ Tags').classes('text-xl font-semibold')
            tags_label = ui.label('Tags will appear here.').classes('w-full')
    
        ui.button('⬇️ Download Summary', on_click=lambda: ui.download.content(
            f"""📄 Summary:
        {summary_label.text}

        🔑 Keywords:
        {keywords_label.text}

        🏷️ Tags:
        {tags_label.text}
        """, 'summary_full.txt')).props('color=primary').classes('w-full')

if __name__ in {'__main__', '__mp_main__'}:
    ui.run(
        host='0.0.0.0',
        port=5001,
        reload=False,
        root_path='/proxy/5001',
        title='PDF Summarizer'
    )