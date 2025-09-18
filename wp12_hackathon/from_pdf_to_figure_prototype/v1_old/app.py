# app_refactored.py (Fixed for older Python versions)
"""
Refactored Flask application with resolved inconsistencies - Python 3.7+ compatible
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import tempfile
import requests
from werkzeug.utils import secure_filename
from pathlib import Path

# Import refactored components
from core.document_processor import DocumentProcessor
from core.database import DatabaseManager
from utils.api_client import LLMClient
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Initialize components with proper dependency injection
db_manager = DatabaseManager(app.config.get('DATABASE_PATH', 'intelligent_rag.db'))
llm_client = LLMClient(app.config['API_KEY'], app.config['BASE_URL'])
processor = DocumentProcessor(llm_client, db_manager)

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload or URL submission with proper error handling"""
    try:
        # Handle file upload
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            if file and file.filename.lower().endswith('.pdf'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                try:
                    # Process document with unified processor
                    results = processor.process_document(filepath)
                    
                    if results.get('success'):
                        return jsonify({
                            'success': True,
                            'document_id': results['document_id'],
                            'redirect': url_for('results', doc_id=results['document_id'])
                        })
                    else:
                        return jsonify({
                            'success': False, 
                            'error': results.get('error', 'Processing failed')
                        })
                        
                finally:
                    # Clean up uploaded file
                    if os.path.exists(filepath):
                        os.remove(filepath)
        
        # Handle URL submission
        elif 'url' in request.form and request.form['url']:
            url = request.form['url']
            
            # Download PDF from URL
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Validate content type
                content_type = response.headers.get('content-type', '')
                if 'pdf' not in content_type.lower():
                    return jsonify({
                        'success': False, 
                        'error': 'URL does not point to a PDF file'
                    })
                
                # Save temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(response.content)
                    tmp_filepath = tmp_file.name
                
                try:
                    # Process document
                    results = processor.process_document(tmp_filepath)
                    
                    if results.get('success'):
                        return jsonify({
                            'success': True,
                            'document_id': results['document_id'],
                            'redirect': url_for('results', doc_id=results['document_id'])
                        })
                    else:
                        return jsonify({
                            'success': False, 
                            'error': results.get('error', 'Processing failed')
                        })
                        
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_filepath):
                        os.unlink(tmp_filepath)
                        
            except requests.exceptions.ConnectionError:
                return jsonify({
                    'success': False, 
                    'error': 'Failed to connect to the URL'
                })
            except requests.exceptions.Timeout:
                return jsonify({
                    'success': False, 
                    'error': 'Request timeout - URL took too long to respond'
                })
            except requests.exceptions.HTTPError as e:
                return jsonify({
                    'success': False, 
                    'error': f'HTTP error: {str(e)}'
                })
            except requests.exceptions.RequestException as e:
                return jsonify({
                    'success': False, 
                    'error': f'Failed to download PDF: {str(e)}'
                })
        
        return jsonify({
            'success': False, 
            'error': 'No valid file or URL provided'
        })
        
    except Exception as e:
        app.logger.error(f"Upload error: {str(e)}")
        return jsonify({
            'success': False, 
            'error': 'An unexpected error occurred during processing'
        })

@app.route('/results/<int:doc_id>')
def results(doc_id):
    """Display analysis results with proper error handling"""
    try:
        intelligence = processor.get_company_intelligence(doc_id)
        
        if 'error' in intelligence:
            return render_template('results.html', 
                                 error=intelligence['error'], 
                                 doc_id=doc_id)
        
        return render_template('results.html', 
                             data=intelligence, 
                             doc_id=doc_id)
                             
    except Exception as e:
        app.logger.error(f"Results error for doc {doc_id}: {str(e)}")
        return render_template('results.html', 
                             error="Failed to load analysis results", 
                             doc_id=doc_id)

@app.route('/api/progress/<int:doc_id>')
def get_progress(doc_id):
    """Get processing progress for real-time updates"""
    try:
        progress = processor.get_processing_progress(doc_id)
        return jsonify(progress)
    except Exception as e:
        app.logger.error(f"Progress error for doc {doc_id}: {str(e)}")
        return jsonify({
            'error': 'Failed to get progress',
            'status': 'error'
        })

@app.route('/api/recent')
def get_recent_analyses():
    """Get recent analyses for dashboard"""
    try:
        recent = processor.get_recent_analyses(limit=6)
        return jsonify(recent)
    except Exception as e:
        app.logger.error(f"Recent analyses error: {str(e)}")
        return jsonify([])

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', 
                         error="Page not found", 
                         code=404), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', 
                         error="Internal server error", 
                         code=500), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)