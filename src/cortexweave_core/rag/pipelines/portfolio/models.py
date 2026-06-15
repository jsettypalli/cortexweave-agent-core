from sqlalchemy import (
    Boolean, Column, Date, DateTime, ForeignKey, Index, Integer,
    JSON, Numeric, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import relationship

from cortexweave_core.rag.db import RAGReadBase


class PortfolioFamily(RAGReadBase):
    __tablename__ = "rag_portfolio_families"

    id = Column(Integer, primary_key=True)
    knowledge_base_id = Column(Integer, ForeignKey("rag_knowledge_bases.id"), nullable=False)
    name = Column(String(255), nullable=False)
    external_id = Column(String(255), nullable=False)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

    reports = relationship("PortfolioReport", back_populates="portfolio_family")

    __table_args__ = (
        UniqueConstraint("knowledge_base_id", "external_id", name="uq_rag_portfolio_family_kb_external_id"),
        Index("idx_rag_portfolio_families_kb_id", "knowledge_base_id"),
    )


class PortfolioReport(RAGReadBase):
    __tablename__ = "rag_portfolio_reports"

    id = Column(Integer, primary_key=True)
    knowledge_base_id = Column(Integer, ForeignKey("rag_knowledge_bases.id"), nullable=False)
    portfolio_family_id = Column(Integer, ForeignKey("rag_portfolio_families.id"), nullable=False)
    source_file_name = Column(String(255), nullable=False)
    holder_name = Column(String(255), nullable=False)
    report_date = Column(Date, nullable=False)
    content_hash = Column(String(64), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

    portfolio_family = relationship("PortfolioFamily", back_populates="reports")
    asset_allocations = relationship("PortfolioAssetAllocation", back_populates="portfolio_report", cascade="all, delete-orphan")
    sub_asset_allocations = relationship("PortfolioSubAssetAllocation", back_populates="portfolio_report", cascade="all, delete-orphan")
    mutual_fund_holdings = relationship("PortfolioMutualFundHolding", back_populates="portfolio_report", cascade="all, delete-orphan")
    pms_holdings = relationship("PortfolioPMSHolding", back_populates="portfolio_report", cascade="all, delete-orphan")
    bond_holdings = relationship("PortfolioBondHolding", back_populates="portfolio_report", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_rag_portfolio_reports_family_date", "portfolio_family_id", "report_date"),
        Index("idx_rag_portfolio_reports_kb_id", "knowledge_base_id"),
        Index("idx_rag_portfolio_reports_content_hash", "content_hash"),
    )


class _AllocationMixin:
    current_value = Column(Numeric(18, 2))
    invested_value = Column(Numeric(18, 2))
    unrealized_gain_loss = Column(Numeric(18, 2))
    dividend_interest_paid = Column(Numeric(18, 2))
    xirr_percent = Column(Numeric(18, 4))
    prescribed_allocation_percent = Column(Numeric(18, 4))
    current_allocation_percent = Column(Numeric(18, 4))
    deviation_percent = Column(Numeric(18, 4))
    total_gain = Column(Numeric(18, 2))
    holding_period_months = Column(Integer)
    realized_gain_loss = Column(Numeric(18, 2))
    source_page = Column(Integer)
    raw_text = Column(Text)


class PortfolioAssetAllocation(RAGReadBase, _AllocationMixin):
    __tablename__ = "rag_portfolio_asset_allocations"
    id = Column(Integer, primary_key=True)
    portfolio_report_id = Column(Integer, ForeignKey("rag_portfolio_reports.id"), nullable=False)
    asset_class = Column(String(255), nullable=False)
    portfolio_report = relationship("PortfolioReport", back_populates="asset_allocations")


class PortfolioSubAssetAllocation(RAGReadBase, _AllocationMixin):
    __tablename__ = "rag_portfolio_sub_asset_allocations"
    id = Column(Integer, primary_key=True)
    portfolio_report_id = Column(Integer, ForeignKey("rag_portfolio_reports.id"), nullable=False)
    asset_class = Column(String(255), nullable=False)
    sub_asset_class = Column(String(255), nullable=False)
    portfolio_report = relationship("PortfolioReport", back_populates="sub_asset_allocations")


class PortfolioMutualFundHolding(RAGReadBase):
    __tablename__ = "rag_portfolio_mutual_fund_holdings"
    id = Column(Integer, primary_key=True)
    portfolio_report_id = Column(Integer, ForeignKey("rag_portfolio_reports.id"), nullable=False)
    scheme_name = Column(String(512), nullable=False)
    asset_class = Column(String(255), nullable=False)
    sub_asset_class = Column(String(255), nullable=False)
    investment_date = Column(Date)
    units = Column(Numeric(18, 4))
    cost = Column(Numeric(18, 2))
    market_value = Column(Numeric(18, 2))
    unrealized_gain_loss = Column(Numeric(18, 2))
    xirr_percent = Column(Numeric(18, 4))
    dividend_paid = Column(Numeric(18, 2))
    total_returns = Column(Numeric(18, 2))
    percent_to_mf_portfolio = Column(Numeric(18, 4))
    purchase_price = Column(Numeric(18, 4))
    nav = Column(Numeric(18, 4))
    gain_loss_percent = Column(Numeric(18, 4))
    holding_period_months = Column(Integer)
    realized_gain_loss = Column(Numeric(18, 2))
    folio_no = Column(String(128))
    source_page = Column(Integer)
    raw_text = Column(Text)
    portfolio_report = relationship("PortfolioReport", back_populates="mutual_fund_holdings")


class PortfolioPMSHolding(RAGReadBase):
    __tablename__ = "rag_portfolio_pms_holdings"
    id = Column(Integer, primary_key=True)
    portfolio_report_id = Column(Integer, ForeignKey("rag_portfolio_reports.id"), nullable=False)
    scheme_name = Column(String(255), nullable=False)
    investment_date = Column(Date)
    investment_amount = Column(Numeric(18, 2))
    current_value = Column(Numeric(18, 2))
    valuation_date = Column(Date)
    unrealized_gain_loss = Column(Numeric(18, 2))
    unrealized_gain_loss_percent = Column(Numeric(18, 4))
    percent_to_pms_portfolio = Column(Numeric(18, 4))
    holding_period_months = Column(Integer)
    source_page = Column(Integer)
    raw_text = Column(Text)
    portfolio_report = relationship("PortfolioReport", back_populates="pms_holdings")


class PortfolioBondHolding(RAGReadBase):
    __tablename__ = "rag_portfolio_bond_holdings"
    id = Column(Integer, primary_key=True)
    portfolio_report_id = Column(Integer, ForeignKey("rag_portfolio_reports.id"), nullable=False)
    security_name = Column(String(255), nullable=False)
    rating = Column(String(64))
    investment_date = Column(Date)
    maturity_date = Column(Date)
    units = Column(Numeric(18, 4))
    cost = Column(Numeric(18, 2))
    market_value = Column(Numeric(18, 2))
    purchase_price = Column(Numeric(18, 4))
    current_price = Column(Numeric(18, 4))
    interest_income = Column(Numeric(18, 2))
    unrealized_gain_loss = Column(Numeric(18, 2))
    unrealized_gain_loss_percent = Column(Numeric(18, 4))
    percent_to_bond_portfolio = Column(Numeric(18, 4))
    holding_period_months = Column(Integer)
    realized_gain_loss = Column(Numeric(18, 2))
    source_page = Column(Integer)
    raw_text = Column(Text)
    portfolio_report = relationship("PortfolioReport", back_populates="bond_holdings")
