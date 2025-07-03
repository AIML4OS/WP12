from nicegui import ui, app
import PyPDF2
from io import BytesIO
import asyncio

# Global variable to store extracted text
extracted_text = ""
text_label = ""

async def handle_upload(e):
    """Handle PDF file upload and extract text"""
    global extracted_text
    
    if not e.content:
        ui.notify('No file selected', type='warning')
        return
    
    try:
        # Create progress indicator
        with ui.dialog() as progress_dialog:
            ui.label('Processing PDF...')
            ui.spinner(size='lg')
        progress_dialog.open()
        
        # Process the PDF
        pdf_reader = PyPDF2.PdfReader(e.content)
        # PyPDF2.PdfReader(BytesIO(e.content))
        text = ""
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += f"--- Page {page_num + 1} ---\n"
            text += page.extract_text() + "\n\n"
        
        extracted_text = text if text.strip() else "No text found in PDF"
        
        # Update the text area
        text_area.value = extracted_text
        text_area.update()
        # text_area.bind_value_from(extracted_text)
        progress_dialog.close()
        ui.notify(f'Successfully extracted text from {len(pdf_reader.pages)} pages', type='positive')
        
    except Exception as error:
        progress_dialog.close()
        extracted_text = f"Error processing PDF: {str(error)}"
        text_area.value = extracted_text
        text_area.update()
        ui.notify('Error processing PDF', type='negative')

def copy_text():
    """Copy extracted text to clipboard"""
    if extracted_text:
        ui.run_javascript(f'navigator.clipboard.writeText(`{extracted_text}`)')
        ui.notify('Text copied to clipboard!', type='positive')
    else:
        ui.notify('No text to copy', type='warning')

def clear_text():
    """Clear the text area"""
    global extracted_text
    extracted_text = ""
    text_area.value = ""
    text_area.update()
    ui.notify('Text cleared', type='info')

# Create the main UI
@ui.page('/')
def main_page():
    global text_area
    
    # Header
    with ui.header(elevated=True).style('background-color: #1976d2'):
        ui.label('PDF Text Summarizer').style('font-size: 1.5rem; font-weight: bold; color: white')
    
    # Main content
    with ui.column().classes('w-full max-w-4xl mx-auto p-4 gap-4'):
        
        # Instructions
        ui.markdown("""
        ## Upload and Extract Text from PDF
        Select a PDF file to extract its text content. The extracted text will appear below.
        """)
        
        # Upload section
        with ui.card().classes('w-full'):
            ui.label('File Upload').classes('text-lg font-semibold mb-2')
            ui.upload(
                on_upload=handle_upload,
                auto_upload=True,
                max_file_size=10_000_000  # 10MB limit
            ).props('accept=".pdf" color="primary"').classes('w-full')
        
        # Settings menu
        with ui.card().classes("w-full"):
            ui.label('Settings Menu').classes('text-lg font-semibold mb-2')
            with ui.grid(columns=2).classes('gap-x-8 gap-y-6 w-full'):
                with ui.row().classes('w-full items-center'):
                    ui.label("Words")
                    slider = ui.slider(min=0, max=100, value=50)
                    ui.label().bind_text_from(slider, 'value')
                with ui.row().classes('w-full items-center'):
                    ui.label("Words")
                    slider = ui.slider(min=0, max=100, value=50)
                    ui.label().bind_text_from(slider, 'value')
                with ui.row().classes('w-full items-center'):
                    ui.label("Words")
                    slider = ui.slider(min=0, max=100, value=50)
                    ui.label().bind_text_from(slider, 'value')

         # Action buttons
        with ui.row().classes('w-full justify-center gap-2'):
            ui.button('Generate Summary', on_click=copy_text, icon='content_copy').props('color="secondary"')

        # Text output section
        with ui.card().classes('w-full'):
            ui.label('Extracted Text').classes('text-lg font-semibold mb-2')
            text_area = ui.textarea(
                placeholder='Extracted text will appear here...',
                value="" # we need this to show properly the value of the text_area
            ).classes('w-full').props('rows="10" outlined readonly')

        # Action buttons
        with ui.row().classes('w-full justify-center gap-2'):
            ui.button('Copy Text', on_click=copy_text, icon='content_copy').props('color="secondary"')
            ui.button('Clear', on_click=clear_text, icon='clear').props('color="negative"')
            ui.button('Download as text file', 
                      on_click=lambda: ui.download.content(extracted_text, 'hello.txt'),
                      icon='download').props(
                          'color="secondary"'
                      )
        ui.add_body_html('<script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>')
        ui.add_head_html('<link href="https://cdn.jsdelivr.net/themify-icons/0.1.2/css/themify-icons.css" rel="stylesheet" />')
        # ui.icon('ti-car').classes('text-5xl')
        ui.button('Files button', on_click=copy_text, icon='ti-files').props('color="secondary"')
        src = 'https://assets5.lottiefiles.com/packages/lf20_MKCnqtNQvg.json'
        ui.html(f'<lottie-player src="{src}" loop autoplay />').classes('w-24')

# Optional: Add FastAPI endpoints
@app.get('/api/health')
def health_check():
    return {'status': 'healthy'}

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(host='127.0.0.1', port=8000, title='PDF Text Summarizer')