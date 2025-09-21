#!/usr/bin/env python3
"""
SRE Analytics Web UI
A Flask-based web interface for generating SRE performance and incident analysis reports.
"""

from flask import Flask, render_template, request, jsonify, send_file, session
import os
import json
import uuid
from datetime import datetime, timedelta
import threading
import logging
from pathlib import Path

# Import our SRE modules
from src.reports.enhanced_sre_report_system import EnhancedSREReportSystem
from src.collectors.oauth_appdynamics_collector import OAuthAppDynamicsCollector

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-key-change-in-production')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global storage for task progress
task_progress = {}

class ReportGenerator:
    """Handle report generation with progress tracking"""

    def __init__(self):
        self.sre_system = None

    def generate_report(self, task_id, report_type, app_name, services, incident_params=None):
        """Generate report with progress tracking"""
        try:
            task_progress[task_id] = {
                'status': 'starting',
                'progress': 0,
                'message': 'Initializing report generation...'
            }

            # Initialize SRE system
            task_progress[task_id].update({
                'progress': 10,
                'message': 'Setting up report system...'
            })

            self.sre_system = EnhancedSREReportSystem(app_name=app_name)

            # Test AppDynamics connection
            task_progress[task_id].update({
                'progress': 20,
                'message': 'Testing AppDynamics connection...'
            })

            collector = OAuthAppDynamicsCollector()
            connection_success = collector.test_connection()

            if connection_success:
                task_progress[task_id].update({
                    'progress': 30,
                    'message': 'Connected to AppDynamics successfully'
                })
            else:
                task_progress[task_id].update({
                    'progress': 30,
                    'message': 'Using demo data (AppDynamics unavailable)'
                })

            # Generate report based on type
            if report_type == 'performance':
                task_progress[task_id].update({
                    'progress': 50,
                    'message': 'Generating performance analysis report...'
                })

                # Check if this is service-level analysis
                analysis_scope = incident_params.get('analysis_scope') if incident_params else None
                target_service = incident_params.get('target_service') if incident_params else None

                if analysis_scope == 'service' and target_service:
                    task_progress[task_id].update({
                        'message': f'Generating service-level analysis for {target_service}...'
                    })
                    # For service-level, focus on the target service
                    report_paths = self.sre_system.generate_full_report_suite(
                        application_name=f"{app_name} - {target_service} Service",
                        services=[target_service] + [s for s in services if s != target_service]
                    )
                else:
                    report_paths = self.sre_system.generate_full_report_suite(
                        application_name=app_name,
                        services=services
                    )

            elif report_type == 'incident':
                task_progress[task_id].update({
                    'progress': 50,
                    'message': 'Generating incident analysis report...'
                })

                # Parse incident time
                incident_time = datetime.now()
                if incident_params and incident_params.get('incident_time'):
                    try:
                        incident_time = datetime.fromisoformat(incident_params['incident_time'])
                    except:
                        incident_time = datetime.now() - timedelta(hours=2)

                duration = float(incident_params.get('duration', 1.0)) if incident_params else 1.0

                # Check if this is service-specific incident analysis
                incident_scope = incident_params.get('incident_scope') if incident_params else None
                target_service = incident_params.get('target_service') if incident_params else None

                if incident_scope == 'service' and target_service:
                    task_progress[task_id].update({
                        'message': f'Analyzing incident impact on {target_service} service...'
                    })
                    # For service-specific, focus on the target service
                    report_paths = self.sre_system.generate_full_report_suite(
                        application_name=f"{app_name} - {target_service} Incident",
                        services=[target_service] + [s for s in services if s != target_service],
                        incident_time=incident_time,
                        incident_duration=duration
                    )
                else:
                    report_paths = self.sre_system.generate_full_report_suite(
                        application_name=app_name,
                        services=services,
                        incident_time=incident_time,
                        incident_duration=duration
                    )

            # Finalize
            task_progress[task_id].update({
                'status': 'completed',
                'progress': 100,
                'message': 'Report generation completed successfully!',
                'report_paths': report_paths
            })

        except Exception as e:
            logger.error(f"Report generation failed for task {task_id}: {e}")
            task_progress[task_id] = {
                'status': 'error',
                'progress': 0,
                'message': f'Report generation failed: {str(e)}'
            }

report_generator = ReportGenerator()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/performance')
def performance_report():
    """Performance analysis report page"""
    return render_template('performance_report.html')

@app.route('/incident')
def incident_report():
    """Incident analysis report page"""
    return render_template('incident_report.html')

@app.route('/api/generate-report', methods=['POST'])
def api_generate_report():
    """API endpoint to generate reports"""
    try:
        data = request.json

        # Validate input
        report_type = data.get('report_type')
        app_name = data.get('app_name', '').strip()
        services_input = data.get('services')

        if not report_type or report_type not in ['performance', 'incident']:
            return jsonify({'error': 'Invalid report type'}), 400

        if not app_name:
            return jsonify({'error': 'Application name is required'}), 400

        if not services_input:
            return jsonify({'error': 'At least one service is required'}), 400

        # Handle both string and list inputs for services
        if isinstance(services_input, list):
            services = [s.strip() for s in services_input if s and str(s).strip()]
        else:
            services = [s.strip() for s in str(services_input).split(',') if s.strip()]

        # Generate unique task ID
        task_id = str(uuid.uuid4())
        session['current_task'] = task_id

        # Parse additional parameters if provided
        incident_params = None
        if report_type == 'incident':
            incident_params = {
                'incident_time': data.get('incident_time'),
                'duration': data.get('duration', 1.0),
                'incident_scope': data.get('incident_scope', 'application'),
                'target_service': data.get('target_service')
            }
        elif report_type == 'performance':
            incident_params = {
                'analysis_scope': data.get('analysis_scope', 'application'),
                'target_service': data.get('target_service')
            }

        # Start report generation in background thread
        thread = threading.Thread(
            target=report_generator.generate_report,
            args=(task_id, report_type, app_name, services, incident_params)
        )
        thread.start()

        return jsonify({
            'task_id': task_id,
            'message': 'Report generation started'
        })

    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/task-status/<task_id>')
def api_task_status(task_id):
    """Get task progress status"""
    status = task_progress.get(task_id, {
        'status': 'not_found',
        'progress': 0,
        'message': 'Task not found'
    })
    return jsonify(status)

@app.route('/api/download-report/<task_id>/<report_type>')
def api_download_report(task_id, report_type):
    """Download generated report"""
    try:
        task_data = task_progress.get(task_id)
        if not task_data or task_data.get('status') != 'completed':
            return jsonify({'error': 'Report not ready'}), 404

        report_paths = task_data.get('report_paths', {})

        if report_type == 'html' and report_paths.get('html_report'):
            return send_file(report_paths['html_report'], as_attachment=True)
        elif report_type == 'pdf' and report_paths.get('pdf_report'):
            return send_file(report_paths['pdf_report'], as_attachment=True)
        elif report_type == 'json' and report_paths.get('json_data'):
            return send_file(report_paths['json_data'], as_attachment=True)
        else:
            return jsonify({'error': 'Report file not found'}), 404

    except Exception as e:
        logger.error(f"Download error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recent-reports')
def api_recent_reports():
    """Get list of recent reports"""
    try:
        reports_dir = Path('reports/generated')
        if not reports_dir.exists():
            return jsonify({'reports': []})

        reports = []
        for file_path in reports_dir.glob('comprehensive_sre_report_*.html'):
            stats = file_path.stat()
            reports.append({
                'name': file_path.name,
                'size': f"{stats.st_size / 1024:.1f} KB",
                'created': datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'path': str(file_path)
            })

        # Sort by creation time (newest first)
        reports.sort(key=lambda x: x['created'], reverse=True)
        return jsonify({'reports': reports[:10]})  # Last 10 reports

    except Exception as e:
        logger.error(f"Recent reports error: {e}")
        return jsonify({'reports': []})

@app.route('/api/appdynamics-applications')
def api_appdynamics_applications():
    """Get list of applications from AppDynamics"""
    try:
        from src.collectors.oauth_appdynamics_collector import OAuthAppDynamicsCollector

        collector = OAuthAppDynamicsCollector()

        # Test connection first
        if not collector.test_connection():
            return jsonify({
                'success': False,
                'error': 'Unable to connect to AppDynamics',
                'applications': []
            })

        # Get applications
        applications = collector.get_applications()

        # Format for UI
        app_list = []
        for app in applications:
            app_list.append({
                'id': app.get('id'),
                'name': app.get('name'),
                'description': app.get('description', '')
            })

        # Sort by name
        app_list.sort(key=lambda x: x['name'].lower())

        return jsonify({
            'success': True,
            'applications': app_list,
            'count': len(app_list)
        })

    except Exception as e:
        logger.error(f"AppDynamics applications API error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'applications': []
        })

if __name__ == '__main__':
    # Ensure reports directory exists
    os.makedirs('reports/generated', exist_ok=True)

    print("🚀 Starting SRE Analytics Web UI...")
    print("📊 Access the interface at: http://localhost:5001")
    print("🔧 Press Ctrl+C to stop")

    app.run(debug=True, host='0.0.0.0', port=5001)