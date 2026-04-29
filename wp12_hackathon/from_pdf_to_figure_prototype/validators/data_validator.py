# validator.py
from typing import Dict, List

class DataValidator:
    @staticmethod
    def validate_and_clean(data: Dict) -> Dict:
        """Validate and clean extracted data"""
        if "error" in data:
            return data

        cleaned = data.copy()
        warnings = []

        # Validate year
        year = cleaned.get("report_year")
        if year and (year < 2020 or year > 2026):
            warnings.append(f"Unusual report year: {year}")

        # Check financial metrics consistency
        financials = cleaned.get("financial_metrics", {})
        revenue = financials.get("total_revenue")
        profit = financials.get("net_profit")

        if revenue and profit:
            if profit > revenue:
                warnings.append("Net profit exceeds revenue - possible data error")
            elif profit > revenue * 0.5:
                warnings.append("Very high profit margin - please verify")

        # Convert very large numbers (probably not in millions)
        for metric in ["total_revenue", "net_profit", "total_assets"]:
            value = financials.get(metric)
            if value and value > 500000:  # Likely in thousands or units
                financials[metric] = value / 1000
                warnings.append(f"{metric} converted from thousands to millions")

        # Validate operational metrics
        ops = cleaned.get("operational_metrics", {})
        employees = ops.get("employees")
        if employees and employees > 10000000:  # 10M employees seems wrong
            warnings.append("Employee count seems unusually high")

        if warnings:
            cleaned["validation_warnings"] = warnings

        return cleaned