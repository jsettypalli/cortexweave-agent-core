from sqlalchemy.orm import Session

from cortexweave_core.rag.pipelines.portfolio.models import PortfolioFamily, PortfolioReport


class PortfolioReportDAO:
    @staticmethod
    def latest_reports_for_family(
        session: Session, knowledge_base_id: int, family_external_id: str
    ) -> list[PortfolioReport]:
        family = session.query(PortfolioFamily).filter_by(
            knowledge_base_id=knowledge_base_id, external_id=family_external_id
        ).first()
        if not family:
            return []
        latest_date = session.query(PortfolioReport.report_date).filter_by(
            portfolio_family_id=family.id, is_active=True,
        ).order_by(PortfolioReport.report_date.desc()).limit(1).scalar()
        if not latest_date:
            return []
        return session.query(PortfolioReport).filter_by(
            portfolio_family_id=family.id, report_date=latest_date, is_active=True,
        ).all()
