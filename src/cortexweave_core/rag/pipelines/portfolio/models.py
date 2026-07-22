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
    fund_resolutions = relationship("PortfolioMFHoldingResolution", back_populates="portfolio_report", cascade="all, delete-orphan")
    enrichment_jobs = relationship("PortfolioEnrichmentJob", back_populates="portfolio_report", cascade="all, delete-orphan")

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
    fund_resolution = relationship("PortfolioMFHoldingResolution", back_populates="mutual_fund_holding", uselist=False, cascade="all, delete-orphan")


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


class PortfolioFundSchemeIndex(RAGReadBase):
    __tablename__ = "rag_portfolio_fund_scheme_index"

    scheme_code = Column(Integer, primary_key=True)
    name = Column(String(512), nullable=False)
    canon_base = Column(String(512), nullable=False, index=True)
    amc = Column(String(255), index=True)
    is_direct = Column(Boolean, default=False, nullable=False)
    category = Column(String(255), index=True)
    fund_house = Column(String(255))
    isin = Column(String(64))
    source = Column(String(64), default="mfapi", nullable=False)
    fetched_at = Column(DateTime)

    __table_args__ = (
        Index("idx_rag_portfolio_scheme_index_canon_base", "canon_base"),
        Index("idx_rag_portfolio_scheme_index_amc", "amc"),
        Index("idx_rag_portfolio_scheme_index_category", "category"),
    )


class PortfolioFundCache(RAGReadBase):
    __tablename__ = "rag_portfolio_fund_cache"

    scheme_code = Column(Integer, primary_key=True)
    name = Column(String(512), nullable=False)
    fund_house = Column(String(255))
    category = Column(String(255))
    scheme_type = Column(String(255))
    isin = Column(String(64))
    groww_search_id = Column(String(255), index=True)
    riskometer = Column(String(255))
    risk_rating = Column(Integer)
    expense_ratio = Column(Numeric(18, 4))
    aum = Column(Numeric(18, 2))
    latest_nav = Column(Numeric(18, 4))
    nav_date = Column(String(64))
    holdings_as_of = Column(String(64))
    meta_fetched_at = Column(DateTime)
    holdings_fetched_at = Column(DateTime)

    holdings = relationship("PortfolioFundCacheHolding", back_populates="fund", cascade="all, delete-orphan")


class PortfolioFundCacheHolding(RAGReadBase):
    __tablename__ = "rag_portfolio_fund_cache_holdings"

    id = Column(Integer, primary_key=True)
    scheme_code = Column(Integer, ForeignKey("rag_portfolio_fund_cache.scheme_code"), nullable=False, index=True)
    stock_name = Column(String(512), nullable=False, index=True)
    sector = Column(String(255), index=True)
    instrument = Column(String(255))
    nature = Column(String(64), index=True)
    pct_of_assets = Column(Numeric(18, 6))
    market_value = Column(Numeric(18, 2))
    rating = Column(String(64))
    snapshot_date = Column(String(64))

    fund = relationship("PortfolioFundCache", back_populates="holdings")


class PortfolioMFHoldingResolution(RAGReadBase):
    __tablename__ = "rag_portfolio_mf_holding_resolutions"

    id = Column(Integer, primary_key=True)
    portfolio_report_id = Column(Integer, ForeignKey("rag_portfolio_reports.id"), nullable=False, index=True)
    mutual_fund_holding_id = Column(Integer, ForeignKey("rag_portfolio_mutual_fund_holdings.id"), nullable=False, unique=True, index=True)
    raw_scheme_name = Column(String(512), nullable=False)
    normalized_scheme_name = Column(String(512), nullable=False)
    statement_category = Column(String(255))
    scheme_code = Column(Integer, ForeignKey("rag_portfolio_fund_scheme_index.scheme_code"))
    matched_name = Column(String(512))
    amc = Column(String(255))
    score = Column(Numeric(18, 4))
    lead = Column(Numeric(18, 4))
    confidence = Column(String(32), nullable=False)
    status = Column(String(32), nullable=False, index=True)
    alternatives_json = Column(JSON, default=list)
    error_message = Column(Text)
    provider = Column(String(64), nullable=False, default="backend_ported_mftracker")
    provider_version = Column(String(64), nullable=False, default="1")
    resolved_at = Column(DateTime)
    metadata_json = Column(JSON, default=dict)

    portfolio_report = relationship("PortfolioReport", back_populates="fund_resolutions")
    mutual_fund_holding = relationship("PortfolioMutualFundHolding", back_populates="fund_resolution")
    security_exposures = relationship("PortfolioMFHoldingSecurityExposure", back_populates="resolution", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_rag_portfolio_resolution_report_status", "portfolio_report_id", "status"),
        Index("idx_rag_portfolio_resolution_scheme_code", "scheme_code"),
    )


class PortfolioMFHoldingSecurityExposure(RAGReadBase):
    __tablename__ = "rag_portfolio_mf_holding_security_exposures"

    id = Column(Integer, primary_key=True)
    resolution_id = Column(Integer, ForeignKey("rag_portfolio_mf_holding_resolutions.id"), nullable=False, index=True)
    mutual_fund_holding_id = Column(Integer, ForeignKey("rag_portfolio_mutual_fund_holdings.id"), nullable=False, index=True)
    scheme_code = Column(Integer, nullable=False, index=True)
    stock_name = Column(String(512), nullable=False, index=True)
    sector = Column(String(255), index=True)
    instrument = Column(String(255))
    nature = Column(String(64), index=True)
    pct_of_fund_assets = Column(Numeric(18, 6))
    fund_current_value = Column(Numeric(18, 2))
    portfolio_weighted_pct = Column(Numeric(18, 6))
    weighted_market_value = Column(Numeric(18, 2))
    source = Column(String(64), default="groww")
    as_of_date = Column(String(64))

    resolution = relationship("PortfolioMFHoldingResolution", back_populates="security_exposures")


class PortfolioEnrichmentJob(RAGReadBase):
    __tablename__ = "rag_portfolio_enrichment_jobs"

    id = Column(Integer, primary_key=True)
    portfolio_report_id = Column(Integer, ForeignKey("rag_portfolio_reports.id"), nullable=False, index=True)
    portfolio_family_id = Column(Integer, ForeignKey("rag_portfolio_families.id"), nullable=False, index=True)
    content_hash = Column(String(64), nullable=False)
    provider = Column(String(64), nullable=False, default="backend_ported_mftracker")
    provider_version = Column(String(64), nullable=False, default="1")
    status = Column(String(32), nullable=False, index=True)
    total_funds = Column(Integer, default=0, nullable=False)
    resolved_count = Column(Integer, default=0, nullable=False)
    ambiguous_count = Column(Integer, default=0, nullable=False)
    not_found_count = Column(Integer, default=0, nullable=False)
    failed_count = Column(Integer, default=0, nullable=False)
    error_message = Column(Text)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

    portfolio_report = relationship("PortfolioReport", back_populates="enrichment_jobs")

    __table_args__ = (
        UniqueConstraint("portfolio_report_id", "content_hash", "provider", "provider_version", name="uq_rag_portfolio_enrichment_job"),
    )
