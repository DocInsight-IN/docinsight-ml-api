import os
from django.apps import apps
from celery import shared_task

from grievance_reports.models import GrievanceReport, Category
from ml_deploy.topic_cluster.data.preprocess import preprocess

def classify_texts(texts):
    ml_deploy_config = apps.get_app_config('ml_deploy')
    sim_model = ml_deploy_config.sim_model
    batch_result = sim_model.predict(texts)
    return batch_result

def preprocess_reports_for_modeling(all_reports):
    texts = [report.subject_content for report in all_reports]
    sentences, token_lists, idx_in = preprocess(texts)
    return (sentences, token_lists, idx_in)

def preprocess_reports_for_sim(all_reports):
    texts = [report.subject_content for report in all_reports]
    return texts

@shared_task
def process_reports_batch():
    batch_size = 128
    reports_to_process = GrievanceReport.objects.filter(classified=False)[:batch_size]
    if not reports_to_process:
        return
    
    preprocessed_reports = preprocess_reports_for_sim(reports_to_process)
    classification_results = classify_texts(preprocessed_reports)

    for report, result in zip(reports_to_process, classification_results):
        best_sim, best_match_node, _, highlighted_html = result
        report.category = Category.objects.get(code=best_match_node.code)
        probability_score = (best_sim - best_sim.min()) / (best_sim.max() - best_sim.min()) * 100
        report.probablity = probability_score
        report.html_content = highlighted_html
        report.classifed = True
        report.save()

@shared_task
def process_report(report_id):
    report = [GrievanceReport.objects.get(id=report_id)]
    if report.classifid:
        return "Report already classified"
    
    report_text = preprocess_reports_for_sim(report)
    results = classify_texts(report_text)

    for report, result in zip(report_text, results):
        best_sim, best_match_node, _, highlighted_html = result
        report.category = Category.objects.get(code=best_match_node.code)
        probability_score = (best_sim - best_sim.min()) / (best_sim.max() - best_sim.min()) * 100
        report.probablity = probability_score
        report.html_content = highlighted_html
        report.classifed = True
        report.save()