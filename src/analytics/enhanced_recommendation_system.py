"""
Enhanced Recommendation System
AI-powered recommendations for multi-source metrics analysis
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

from .generic_metrics_engine import AnalysisResult, SLOResult, SLOTarget
from ..data_sources.base import StandardMetric, MetricType


@dataclass
class Recommendation:
    """Enhanced recommendation structure"""
    category: str  # "performance", "reliability", "scalability", "cost", "security"
    priority: str  # "critical", "high", "medium", "low"
    title: str
    description: str
    action_items: List[str]
    expected_impact: str
    implementation_effort: str  # "low", "medium", "high"
    time_to_implement: str
    metrics_affected: List[str]
    confidence_score: float  # 0.0 to 1.0
    supporting_evidence: List[str]


@dataclass
class RecommendationContext:
    """Context for generating recommendations"""
    service_name: str
    analysis_result: AnalysisResult
    historical_data: List[StandardMetric]
    business_context: Dict[str, Any]  # SLA requirements, business criticality, etc.


class EnhancedRecommendationSystem:
    """Enhanced AI-powered recommendation system"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
        self.knowledge_base = self._initialize_knowledge_base()

    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """Initialize knowledge base with best practices and patterns"""
        return {
            "performance_patterns": {
                "high_response_time": {
                    "common_causes": ["database queries", "external API calls", "inefficient algorithms", "resource contention"],
                    "solutions": ["caching", "database optimization", "async processing", "load balancing"],
                    "monitoring": ["response time percentiles", "database query performance", "CPU/memory usage"]
                },
                "high_error_rate": {
                    "common_causes": ["application bugs", "external service failures", "resource exhaustion", "configuration issues"],
                    "solutions": ["circuit breakers", "retry mechanisms", "error handling", "monitoring"],
                    "monitoring": ["error logs", "dependency health", "resource utilization"]
                }
            },
            "scalability_thresholds": {
                "cpu_utilization": {"warning": 70, "critical": 85},
                "memory_utilization": {"warning": 75, "critical": 90},
                "response_time": {"warning": 200, "critical": 500},
                "error_rate": {"warning": 1, "critical": 5}
            },
            "technology_recommendations": {
                "caching": {
                    "redis": {"use_cases": ["session storage", "frequently accessed data"], "complexity": "low"},
                    "memcached": {"use_cases": ["simple key-value caching"], "complexity": "low"},
                    "cdn": {"use_cases": ["static content", "geographic distribution"], "complexity": "medium"}
                },
                "load_balancing": {
                    "nginx": {"use_cases": ["http load balancing", "reverse proxy"], "complexity": "low"},
                    "haproxy": {"use_cases": ["tcp/http load balancing"], "complexity": "medium"},
                    "cloud_lb": {"use_cases": ["cloud-native applications"], "complexity": "low"}
                }
            }
        }

    def generate_comprehensive_recommendations(self, contexts: List[RecommendationContext]) -> Dict[str, List[Recommendation]]:
        """Generate comprehensive recommendations for multiple services"""
        all_recommendations = {}

        for context in contexts:
            try:
                service_recommendations = self._generate_service_recommendations(context)
                all_recommendations[context.service_name] = service_recommendations
            except Exception as e:
                self.logger.error(f"Error generating recommendations for {context.service_name}: {e}")
                all_recommendations[context.service_name] = []

        # Generate cross-service recommendations
        cross_service_recommendations = self._generate_cross_service_recommendations(contexts)
        if cross_service_recommendations:
            all_recommendations["system_wide"] = cross_service_recommendations

        return all_recommendations

    def _generate_service_recommendations(self, context: RecommendationContext) -> List[Recommendation]:
        """Generate recommendations for a single service"""
        recommendations = []

        # Analyze each SLO result
        for slo_result in context.analysis_result.slo_results:
            if slo_result.status in ["breached", "at_risk"]:
                rec = self._generate_metric_specific_recommendation(slo_result, context)
                if rec:
                    recommendations.append(rec)

        # Generate proactive recommendations
        proactive_recs = self._generate_proactive_recommendations(context)
        recommendations.extend(proactive_recs)

        # Use AI for advanced recommendations if available
        if self.llm_client:
            ai_recs = self._generate_ai_recommendations(context)
            recommendations.extend(ai_recs)

        # Sort by priority and confidence
        recommendations.sort(key=lambda x: (
            self._priority_score(x.priority),
            -x.confidence_score
        ))

        return recommendations

    def _generate_metric_specific_recommendation(self, slo_result: SLOResult,
                                               context: RecommendationContext) -> Optional[Recommendation]:
        """Generate specific recommendation based on SLO violation"""
        try:
            metric_type = slo_result.metric_type
            service_name = context.service_name

            if metric_type == MetricType.RESPONSE_TIME:
                return self._recommend_response_time_optimization(slo_result, context)
            elif metric_type == MetricType.ERROR_RATE:
                return self._recommend_error_rate_reduction(slo_result, context)
            elif metric_type == MetricType.CPU_UTILIZATION:
                return self._recommend_cpu_optimization(slo_result, context)
            elif metric_type == MetricType.MEMORY_UTILIZATION:
                return self._recommend_memory_optimization(slo_result, context)
            elif metric_type == MetricType.AVAILABILITY:
                return self._recommend_availability_improvement(slo_result, context)
            else:
                return self._recommend_generic_optimization(slo_result, context)

        except Exception as e:
            self.logger.error(f"Error generating metric-specific recommendation: {e}")
            return None

    def _recommend_response_time_optimization(self, slo_result: SLOResult,
                                            context: RecommendationContext) -> Recommendation:
        """Recommend response time optimization"""
        current_value = slo_result.current_value
        target_value = slo_result.target_value
        excess_latency = current_value - target_value

        priority = "critical" if excess_latency > target_value else "high"
        confidence = 0.8 if slo_result.trend == "degrading" else 0.6

        action_items = [
            "Implement Redis caching for frequently accessed data",
            "Optimize database queries and add appropriate indexes",
            "Consider implementing CDN for static content"
        ]

        if current_value > 500:
            action_items.extend([
                "Review application architecture for bottlenecks",
                "Consider horizontal scaling or load balancing"
            ])

        return Recommendation(
            category="performance",
            priority=priority,
            title=f"Optimize Response Time for {context.service_name}",
            description=f"Response time is {current_value:.1f}ms, exceeding target of {target_value:.1f}ms by {excess_latency:.1f}ms",
            action_items=action_items,
            expected_impact=f"Reduce response time by 30-50% to achieve target",
            implementation_effort="medium",
            time_to_implement="1-2 weeks",
            metrics_affected=["response_time", "user_satisfaction", "throughput"],
            confidence_score=confidence,
            supporting_evidence=[
                f"Current response time: {current_value:.1f}ms",
                f"Target: {target_value:.1f}ms",
                f"Trend: {slo_result.trend}"
            ]
        )

    def _recommend_error_rate_reduction(self, slo_result: SLOResult,
                                      context: RecommendationContext) -> Recommendation:
        """Recommend error rate reduction"""
        current_rate = slo_result.current_value
        target_rate = slo_result.target_value

        priority = "critical" if current_rate > 5 else "high"
        confidence = 0.9

        action_items = [
            "Implement circuit breaker pattern for external dependencies",
            "Add comprehensive error handling and retry logic",
            "Set up detailed error monitoring and alerting"
        ]

        if current_rate > 10:
            action_items.extend([
                "Conduct immediate root cause analysis",
                "Review recent deployments for potential issues"
            ])

        return Recommendation(
            category="reliability",
            priority=priority,
            title=f"Reduce Error Rate for {context.service_name}",
            description=f"Error rate is {current_rate:.2f}%, exceeding target of {target_rate:.2f}%",
            action_items=action_items,
            expected_impact="Improve service reliability and user experience",
            implementation_effort="medium",
            time_to_implement="1-3 weeks",
            metrics_affected=["error_rate", "availability", "user_satisfaction"],
            confidence_score=confidence,
            supporting_evidence=[
                f"Current error rate: {current_rate:.2f}%",
                f"Target: {target_rate:.2f}%",
                f"Error budget consumed: {slo_result.error_budget_consumed:.1f}%"
            ]
        )

    def _recommend_cpu_optimization(self, slo_result: SLOResult,
                                  context: RecommendationContext) -> Recommendation:
        """Recommend CPU optimization"""
        current_cpu = slo_result.current_value
        target_cpu = slo_result.target_value

        priority = "critical" if current_cpu > 90 else "high"
        confidence = 0.7

        action_items = [
            "Profile application to identify CPU-intensive operations",
            "Optimize algorithms and reduce computational complexity",
            "Consider horizontal scaling to distribute load"
        ]

        if current_cpu > 85:
            action_items.extend([
                "Implement auto-scaling policies",
                "Review resource allocation and container limits"
            ])

        return Recommendation(
            category="scalability",
            priority=priority,
            title=f"Optimize CPU Usage for {context.service_name}",
            description=f"CPU utilization is {current_cpu:.1f}%, exceeding target of {target_cpu:.1f}%",
            action_items=action_items,
            expected_impact="Improve system responsiveness and prevent resource exhaustion",
            implementation_effort="medium",
            time_to_implement="2-4 weeks",
            metrics_affected=["cpu_utilization", "response_time", "throughput"],
            confidence_score=confidence,
            supporting_evidence=[
                f"Current CPU usage: {current_cpu:.1f}%",
                f"Target: {target_cpu:.1f}%",
                f"Trend: {slo_result.trend}"
            ]
        )

    def _recommend_memory_optimization(self, slo_result: SLOResult,
                                     context: RecommendationContext) -> Recommendation:
        """Recommend memory optimization"""
        current_memory = slo_result.current_value
        target_memory = slo_result.target_value

        priority = "critical" if current_memory > 95 else "high"
        confidence = 0.8

        action_items = [
            "Profile memory usage to identify memory leaks",
            "Optimize data structures and caching strategies",
            "Review garbage collection settings"
        ]

        if current_memory > 90:
            action_items.extend([
                "Increase memory allocation or scale horizontally",
                "Implement memory monitoring and alerting"
            ])

        return Recommendation(
            category="scalability",
            priority=priority,
            title=f"Optimize Memory Usage for {context.service_name}",
            description=f"Memory utilization is {current_memory:.1f}%, exceeding target of {target_memory:.1f}%",
            action_items=action_items,
            expected_impact="Prevent out-of-memory errors and improve stability",
            implementation_effort="medium",
            time_to_implement="1-3 weeks",
            metrics_affected=["memory_utilization", "availability", "performance"],
            confidence_score=confidence,
            supporting_evidence=[
                f"Current memory usage: {current_memory:.1f}%",
                f"Target: {target_memory:.1f}%",
                f"Risk of memory exhaustion"
            ]
        )

    def _recommend_availability_improvement(self, slo_result: SLOResult,
                                          context: RecommendationContext) -> Recommendation:
        """Recommend availability improvement"""
        current_availability = slo_result.current_value
        target_availability = slo_result.target_value

        priority = "critical"
        confidence = 0.9

        action_items = [
            "Implement health checks and readiness probes",
            "Set up automated failover and recovery mechanisms",
            "Review and improve deployment strategies (blue-green, canary)"
        ]

        return Recommendation(
            category="reliability",
            priority=priority,
            title=f"Improve Availability for {context.service_name}",
            description=f"Availability is {current_availability:.2f}%, below target of {target_availability:.2f}%",
            action_items=action_items,
            expected_impact="Ensure service meets availability SLA requirements",
            implementation_effort="high",
            time_to_implement="2-6 weeks",
            metrics_affected=["availability", "error_rate", "user_satisfaction"],
            confidence_score=confidence,
            supporting_evidence=[
                f"Current availability: {current_availability:.2f}%",
                f"Target: {target_availability:.2f}%",
                f"Downtime impact on users"
            ]
        )

    def _recommend_generic_optimization(self, slo_result: SLOResult,
                                      context: RecommendationContext) -> Recommendation:
        """Generate generic recommendation for custom metrics"""
        return Recommendation(
            category="performance",
            priority="medium",
            title=f"Optimize {slo_result.metric_type.value} for {context.service_name}",
            description=f"Metric {slo_result.metric_type.value} requires attention",
            action_items=[
                "Investigate metric trends and patterns",
                "Review monitoring and alerting setup",
                "Consider metric-specific optimizations"
            ],
            expected_impact="Improve overall service performance",
            implementation_effort="low",
            time_to_implement="1 week",
            metrics_affected=[slo_result.metric_type.value],
            confidence_score=0.5,
            supporting_evidence=[f"SLO compliance issue detected"]
        )

    def _generate_proactive_recommendations(self, context: RecommendationContext) -> List[Recommendation]:
        """Generate proactive recommendations based on trends"""
        recommendations = []

        # Check for degrading trends
        degrading_metrics = [slo for slo in context.analysis_result.slo_results if slo.trend == "degrading"]

        if degrading_metrics:
            rec = Recommendation(
                category="reliability",
                priority="medium",
                title=f"Address Degrading Trends in {context.service_name}",
                description=f"Detected {len(degrading_metrics)} metrics with degrading trends",
                action_items=[
                    "Set up predictive alerting for trend analysis",
                    "Investigate root causes of performance degradation",
                    "Implement preventive measures before SLO breaches"
                ],
                expected_impact="Prevent future SLO violations",
                implementation_effort="low",
                time_to_implement="1 week",
                metrics_affected=[m.metric_type.value for m in degrading_metrics],
                confidence_score=0.7,
                supporting_evidence=[f"Trends indicate potential future issues"]
            )
            recommendations.append(rec)

        return recommendations

    def _generate_cross_service_recommendations(self, contexts: List[RecommendationContext]) -> List[Recommendation]:
        """Generate system-wide recommendations"""
        recommendations = []

        # Analyze patterns across services
        all_slo_results = []
        for context in contexts:
            all_slo_results.extend(context.analysis_result.slo_results)

        # Check for common issues across services
        common_issues = self._identify_common_issues(all_slo_results)

        for issue_type, affected_services in common_issues.items():
            if len(affected_services) > 1:
                rec = self._generate_system_wide_recommendation(issue_type, affected_services)
                if rec:
                    recommendations.append(rec)

        return recommendations

    def _identify_common_issues(self, slo_results: List[SLOResult]) -> Dict[str, List[str]]:
        """Identify common issues across services"""
        issues = {}

        for slo in slo_results:
            if slo.status in ["breached", "at_risk"]:
                issue_key = f"{slo.metric_type.value}_issues"
                if issue_key not in issues:
                    issues[issue_key] = []
                issues[issue_key].append(slo.service_name)

        return issues

    def _generate_system_wide_recommendation(self, issue_type: str, affected_services: List[str]) -> Optional[Recommendation]:
        """Generate system-wide recommendation"""
        if "response_time" in issue_type:
            return Recommendation(
                category="performance",
                priority="high",
                title="System-wide Response Time Optimization",
                description=f"Response time issues detected across {len(affected_services)} services",
                action_items=[
                    "Implement centralized caching strategy",
                    "Review network infrastructure and latency",
                    "Consider implementing API gateway with rate limiting"
                ],
                expected_impact="Improve overall system performance",
                implementation_effort="high",
                time_to_implement="4-8 weeks",
                metrics_affected=["response_time", "throughput"],
                confidence_score=0.8,
                supporting_evidence=[f"Affected services: {', '.join(affected_services)}"]
            )

        return None

    def _generate_ai_recommendations(self, context: RecommendationContext) -> List[Recommendation]:
        """Generate AI-powered recommendations"""
        if not self.llm_client:
            return []

        try:
            # Prepare context for AI analysis
            ai_context = self._prepare_ai_context(context)

            # Generate AI recommendations
            ai_response = self._query_ai_for_recommendations(ai_context)

            # Parse AI response into recommendation objects
            ai_recommendations = self._parse_ai_recommendations(ai_response, context)

            return ai_recommendations

        except Exception as e:
            self.logger.error(f"Error generating AI recommendations: {e}")
            return []

    def _prepare_ai_context(self, context: RecommendationContext) -> str:
        """Prepare context for AI analysis"""
        slo_summary = []
        for slo in context.analysis_result.slo_results:
            slo_summary.append(
                f"- {slo.metric_type.value}: {slo.current_value} (target: {slo.target_value}, "
                f"status: {slo.status}, trend: {slo.trend})"
            )

        return f"""
        Service: {context.service_name}
        Overall Health: {context.analysis_result.overall_health}

        SLO Results:
        {chr(10).join(slo_summary)}

        Current Insights:
        {chr(10).join(context.analysis_result.key_insights)}

        Please provide specific, actionable recommendations for improving this service's performance and reliability.
        """

    def _query_ai_for_recommendations(self, ai_context: str) -> str:
        """Query AI for recommendations"""
        # This would interface with the LLM client
        # Implementation depends on the specific LLM API being used
        return "AI recommendation placeholder"

    def _parse_ai_recommendations(self, ai_response: str, context: RecommendationContext) -> List[Recommendation]:
        """Parse AI response into recommendation objects"""
        # Parse AI response and convert to Recommendation objects
        # This is a placeholder implementation
        return []

    def _priority_score(self, priority: str) -> int:
        """Convert priority to numeric score for sorting"""
        priority_map = {"critical": 1, "high": 2, "medium": 3, "low": 4}
        return priority_map.get(priority, 5)

    def export_recommendations(self, recommendations: Dict[str, List[Recommendation]],
                             format: str = "json") -> str:
        """Export recommendations in specified format"""
        try:
            if format.lower() == "json":
                import json
                export_data = {}
                for service, recs in recommendations.items():
                    export_data[service] = [rec.__dict__ for rec in recs]
                return json.dumps(export_data, indent=2, default=str)
            elif format.lower() == "markdown":
                return self._export_as_markdown(recommendations)
            else:
                raise ValueError(f"Unsupported export format: {format}")
        except Exception as e:
            self.logger.error(f"Error exporting recommendations: {e}")
            return ""

    def _export_as_markdown(self, recommendations: Dict[str, List[Recommendation]]) -> str:
        """Export recommendations as Markdown"""
        lines = ["# Service Recommendations\n"]

        for service_name, recs in recommendations.items():
            lines.append(f"## {service_name}\n")

            for rec in recs:
                lines.extend([
                    f"### {rec.title}",
                    f"**Priority:** {rec.priority}",
                    f"**Category:** {rec.category}",
                    f"**Description:** {rec.description}",
                    "",
                    "**Action Items:**",
                    *[f"- {item}" for item in rec.action_items],
                    "",
                    f"**Expected Impact:** {rec.expected_impact}",
                    f"**Implementation Effort:** {rec.implementation_effort}",
                    f"**Time to Implement:** {rec.time_to_implement}",
                    f"**Confidence Score:** {rec.confidence_score:.1f}",
                    ""
                ])

        return "\n".join(lines)