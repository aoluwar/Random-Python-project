"""
Personal Finance Tracker - Streamlit Web Application

A modern, user-friendly web interface for managing personal finances.
Built with Streamlit for rapid deployment and excellent user experience.

To run this app:
1. Install dependencies: pip install streamlit pandas plotly
2. Run: streamlit run finance_app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
from decimal import Decimal
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional

# Import our finance tracker classes
# (In a real app, these would be in separate modules)
from enum import Enum
from dataclasses import dataclass, asdict
from contextlib import contextmanager


# Configure page settings
st.set_page_config(
    page_title="Personal Finance Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    
    .positive-metric {
        border-left-color: #2e7d32;
    }
    
    .negative-metric {
        border-left-color: #d32f2f;
    }
    
    .sidebar .sidebar-content {
        background-color: #fafafa;
    }
    
    .stSelectbox > div > div > div {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)


class TransactionType(Enum):
    """Transaction type enumeration"""
    INCOME = "income"
    EXPENSE = "expense"
    TRANSFER = "transfer"


class Category(Enum):
    """Category enumeration for transactions"""
    # Income categories
    SALARY = "salary"
    FREELANCE = "freelance"
    INVESTMENT = "investment"
    BONUS = "bonus"
    OTHER_INCOME = "other_income"
    
    # Expense categories
    GROCERIES = "groceries"
    UTILITIES = "utilities"
    TRANSPORTATION = "transportation"
    ENTERTAINMENT = "entertainment"
    HEALTHCARE = "healthcare"
    DINING = "dining_out"
    SHOPPING = "shopping"
    RENT = "rent"
    INSURANCE = "insurance"
    EDUCATION = "education"
    MISCELLANEOUS = "miscellaneous"


@dataclass
class Transaction:
    """Transaction data model"""
    amount: Decimal
    transaction_type: TransactionType
    category: Category
    description: str
    transaction_date: date
    transaction_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.transaction_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.transaction_id = f"txn_{timestamp}"
    
    @property
    def formatted_amount(self) -> str:
        return f"${self.amount:.2f}"
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['transaction_type'] = self.transaction_type.value
        data['category'] = self.category.value
        data['transaction_date'] = self.transaction_date.isoformat()
        data['amount'] = str(self.amount)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Transaction':
        return cls(
            amount=Decimal(data['amount']),
            transaction_type=TransactionType(data['transaction_type']),
            category=Category(data['category']),
            description=data['description'],
            transaction_date=date.fromisoformat(data['transaction_date']),
            transaction_id=data.get('transaction_id')
        )


class StreamlitFinanceTracker:
    """Streamlit-optimized finance tracker with session state management"""
    
    def __init__(self):
        self.data_file = Path.home() / '.finance_tracker_streamlit' / 'transactions.json'
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'transactions' not in st.session_state:
            st.session_state.transactions = self.load_transactions()
        if 'last_saved' not in st.session_state:
            st.session_state.last_saved = datetime.now()
    
    def load_transactions(self) -> List[Transaction]:
        """Load transactions from file"""
        if not self.data_file.exists():
            return []
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
            return [Transaction.from_dict(item) for item in data]
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return []
    
    def save_transactions(self):
        """Save transactions to file"""
        try:
            data = [transaction.to_dict() for transaction in st.session_state.transactions]
            with open(self.data_file, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=2)
            st.session_state.last_saved = datetime.now()
            return True
        except Exception as e:
            st.error(f"Error saving data: {e}")
            return False
    
    def add_transaction(self, amount: Decimal, transaction_type: TransactionType, 
                       category: Category, description: str, transaction_date: date):
        """Add new transaction"""
        transaction = Transaction(
            amount=amount,
            transaction_type=transaction_type,
            category=category,
            description=description,
            transaction_date=transaction_date
        )
        
        st.session_state.transactions.append(transaction)
        self.save_transactions()
        return transaction
    
    def delete_transaction(self, transaction_id: str) -> bool:
        """Delete transaction by ID"""
        original_count = len(st.session_state.transactions)
        st.session_state.transactions = [
            t for t in st.session_state.transactions 
            if t.transaction_id != transaction_id
        ]
        
        if len(st.session_state.transactions) < original_count:
            self.save_transactions()
            return True
        return False
    
    def get_transactions_df(self) -> pd.DataFrame:
        """Convert transactions to pandas DataFrame for analysis"""
        if not st.session_state.transactions:
            return pd.DataFrame()
        
        data = []
        for t in st.session_state.transactions:
            data.append({
                'Date': t.transaction_date,
                'Type': t.transaction_type.value.title(),
                'Category': t.category.value.replace('_', ' ').title(),
                'Amount': float(t.amount),
                'Description': t.description,
                'ID': t.transaction_id
            })
        
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        return df.sort_values('Date', ascending=False)
    
    def calculate_metrics(self) -> Dict:
        """Calculate financial metrics"""
        df = self.get_transactions_df()
        
        if df.empty:
            return {
                'total_income': 0,
                'total_expenses': 0,
                'net_worth': 0,
                'transaction_count': 0
            }
        
        income_mask = df['Type'] == 'Income'
        expense_mask = df['Type'] == 'Expense'
        
        total_income = df[income_mask]['Amount'].sum()
        total_expenses = df[expense_mask]['Amount'].sum()
        
        return {
            'total_income': total_income,
            'total_expenses': total_expenses,
            'net_worth': total_income - total_expenses,
            'transaction_count': len(df)
        }


def render_sidebar():
    """Render the sidebar navigation"""
    st.sidebar.title("💰 Finance Tracker")
    st.sidebar.markdown("---")
    
    pages = {
        "📊 Dashboard": "dashboard",
        "➕ Add Transaction": "add_transaction",
        "📋 View Transactions": "view_transactions",
        "📈 Analytics": "analytics",
        "⚙️ Settings": "settings"
    }
    
    selected_page = st.sidebar.radio("Navigate to:", list(pages.keys()))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Stats")
    
    # Display quick metrics in sidebar
    metrics = tracker.calculate_metrics()
    st.sidebar.metric("Net Worth", f"${metrics['net_worth']:,.2f}")
    st.sidebar.metric("Total Income", f"${metrics['total_income']:,.2f}")
    st.sidebar.metric("Total Expenses", f"${metrics['total_expenses']:,.2f}")
    
    return pages[selected_page]


def render_dashboard():
    """Render the main dashboard"""
    st.title("📊 Financial Dashboard")
    
    metrics = tracker.calculate_metrics()
    df = tracker.get_transactions_df()
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Net Worth",
            value=f"${metrics['net_worth']:,.2f}",
            delta=f"${metrics['net_worth']:,.2f}" if metrics['net_worth'] >= 0 else None
        )
    
    with col2:
        st.metric(
            label="Total Income",
            value=f"${metrics['total_income']:,.2f}",
            delta=f"+${metrics['total_income']:,.2f}"
        )
    
    with col3:
        st.metric(
            label="Total Expenses",
            value=f"${metrics['total_expenses']:,.2f}",
            delta=f"-${metrics['total_expenses']:,.2f}"
        )
    
    with col4:
        st.metric(
            label="Transactions",
            value=metrics['transaction_count']
        )
    
    if not df.empty:
        st.markdown("---")
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💳 Spending by Category")
            expense_df = df[df['Type'] == 'Expense']
            if not expense_df.empty:
                category_spending = expense_df.groupby('Category')['Amount'].sum().reset_index()
                fig_pie = px.pie(
                    category_spending, 
                    values='Amount', 
                    names='Category',
                    title="Expense Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No expense data available for chart")
        
        with col2:
            st.subheader("📈 Monthly Trends")
            if len(df) > 1:
                df['Month'] = df['Date'].dt.to_period('M').astype(str)
                monthly_data = df.groupby(['Month', 'Type'])['Amount'].sum().reset_index()
                
                fig_bar = px.bar(
                    monthly_data,
                    x='Month',
                    y='Amount',
                    color='Type',
                    title="Monthly Income vs Expenses",
                    barmode='group',
                    color_discrete_map={'Income': '#2e7d32', 'Expense': '#d32f2f'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Need more data for trend analysis")
        
        # Recent transactions
        st.markdown("---")
        st.subheader("🕒 Recent Transactions")
        recent_df = df.head(5)
        st.dataframe(
            recent_df[['Date', 'Type', 'Category', 'Amount', 'Description']],
            use_container_width=True
        )
    
    else:
        st.info("No transactions yet. Use the sidebar to add your first transaction!")


def render_add_transaction():
    """Render the add transaction form"""
    st.title("➕ Add New Transaction")
    
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            transaction_type = st.selectbox(
                "Transaction Type",
                options=[t.value.title() for t in TransactionType],
                help="Choose whether this is income or an expense"
            )
            
            amount = st.number_input(
                "Amount ($)",
                min_value=0.01,
                format="%.2f",
                help="Enter the transaction amount"
            )
            
            transaction_date = st.date_input(
                "Date",
                value=date.today(),
                help="When did this transaction occur?"
            )
        
        with col2:
            # Filter categories based on transaction type
            if transaction_type.lower() == 'income':
                category_options = [c for c in Category if c.value in 
                                 ['salary', 'freelance', 'investment', 'bonus', 'other_income']]
            else:
                category_options = [c for c in Category if c.value not in 
                                 ['salary', 'freelance', 'investment', 'bonus', 'other_income']]
            
            category = st.selectbox(
                "Category",
                options=[c.value.replace('_', ' ').title() for c in category_options],
                help="Select the appropriate category"
            )
            
            description = st.text_input(
                "Description",
                placeholder="Enter a brief description...",
                help="Optional: Add details about this transaction"
            )
        
        submitted = st.form_submit_button(
            "Add Transaction",
            type="primary",
            use_container_width=True
        )
        
        if submitted:
            if amount > 0:
                # Convert back to enums
                transaction_type_enum = TransactionType(transaction_type.lower())
                category_enum = Category(category.lower().replace(' ', '_'))
                
                transaction = tracker.add_transaction(
                    amount=Decimal(str(amount)),
                    transaction_type=transaction_type_enum,
                    category=category_enum,
                    description=description or f"{category} transaction",
                    transaction_date=transaction_date
                )
                
                st.success(f"✅ Added {transaction_type.lower()}: ${amount:.2f} - {description}")
                st.balloons()
                
                # Option to add another transaction
                if st.button("Add Another Transaction"):
                    st.rerun()
            else:
                st.error("Please enter a valid amount greater than 0")


def render_view_transactions():
    """Render the transactions view with filtering and editing"""
    st.title("📋 View Transactions")
    
    df = tracker.get_transactions_df()
    
    if df.empty:
        st.info("No transactions to display. Add some transactions first!")
        return
    
    # Filters
    st.subheader("🔍 Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        type_filter = st.multiselect(
            "Transaction Type",
            options=df['Type'].unique(),
            default=df['Type'].unique()
        )
    
    with col2:
        category_filter = st.multiselect(
            "Category",
            options=sorted(df['Category'].unique()),
            default=df['Category'].unique()
        )
    
    with col3:
        date_range = st.date_input(
            "Date Range",
            value=(df['Date'].min().date(), df['Date'].max().date()),
            help="Filter by date range"
        )
    
    # Apply filters
    filtered_df = df[
        (df['Type'].isin(type_filter)) &
        (df['Category'].isin(category_filter))
    ]
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['Date'].dt.date >= start_date) &
            (filtered_df['Date'].dt.date <= end_date)
        ]
    
    # Display results
    st.markdown("---")
    st.subheader(f"📊 Showing {len(filtered_df)} transactions")
    
    if not filtered_df.empty:
        # Summary metrics for filtered data
        col1, col2, col3 = st.columns(3)
        with col1:
            total_income = filtered_df[filtered_df['Type'] == 'Income']['Amount'].sum()
            st.metric("Filtered Income", f"${total_income:,.2f}")
        with col2:
            total_expenses = filtered_df[filtered_df['Type'] == 'Expense']['Amount'].sum()
            st.metric("Filtered Expenses", f"${total_expenses:,.2f}")
        with col3:
            net_amount = total_income - total_expenses
            st.metric("Net Amount", f"${net_amount:,.2f}")
        
        # Transactions table with delete functionality
        for idx, row in filtered_df.iterrows():
            with st.container():
                col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 2, 1, 3, 1])
                
                with col1:
                    st.write(f"**{row['Date'].strftime('%Y-%m-%d')}**")
                with col2:
                    color = "green" if row['Type'] == 'Income' else "red"
                    st.markdown(f"<span style='color: {color}'>{row['Type']}</span>", 
                              unsafe_allow_html=True)
                with col3:
                    st.write(row['Category'])
                with col4:
                    st.write(f"**${row['Amount']:,.2f}**")
                with col5:
                    st.write(row['Description'])
                with col6:
                    if st.button("🗑️", key=f"delete_{row['ID']}", help="Delete transaction"):
                        if tracker.delete_transaction(row['ID']):
                            st.success("Transaction deleted!")
                            st.rerun()
                        else:
                            st.error("Failed to delete transaction")
                
                st.divider()


def render_analytics():
    """Render advanced analytics and insights"""
    st.title("📈 Financial Analytics")
    
    df = tracker.get_transactions_df()
    
    if df.empty:
        st.info("No data available for analytics. Add some transactions first!")
        return
    
    # Time period selector
    col1, col2 = st.columns(2)
    with col1:
        analysis_period = st.selectbox(
            "Analysis Period",
            options=['All Time', 'Last 30 Days', 'Last 90 Days', 'This Year'],
            index=0
        )
    
    # Filter data based on period
    if analysis_period == 'Last 30 Days':
        cutoff_date = datetime.now() - timedelta(days=30)
        df = df[df['Date'] >= cutoff_date]
    elif analysis_period == 'Last 90 Days':
        cutoff_date = datetime.now() - timedelta(days=90)
        df = df[df['Date'] >= cutoff_date]
    elif analysis_period == 'This Year':
        df = df[df['Date'].dt.year == datetime.now().year]
    
    if df.empty:
        st.warning(f"No data available for {analysis_period}")
        return
    
    # Analytics sections
    st.markdown("---")
    
    # 1. Spending patterns
    st.subheader("💳 Spending Patterns")
    
    expense_df = df[df['Type'] == 'Expense']
    if not expense_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top spending categories
            top_categories = expense_df.groupby('Category')['Amount'].sum().sort_values(ascending=False).head(5)
            fig_bar = px.bar(
                x=top_categories.index,
                y=top_categories.values,
                title="Top 5 Spending Categories",
                labels={'x': 'Category', 'y': 'Amount ($)'},
                color=top_categories.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Average transaction amount by category
            avg_by_category = expense_df.groupby('Category')['Amount'].mean().sort_values(ascending=False)
            fig_bar2 = px.bar(
                x=avg_by_category.index,
                y=avg_by_category.values,
                title="Average Transaction Amount by Category",
                labels={'x': 'Category', 'y': 'Average Amount ($)'},
                color=avg_by_category.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_bar2, use_container_width=True)
    
    # 2. Time series analysis
    st.subheader("📊 Time Series Analysis")
    
    if len(df) > 5:  # Need sufficient data for meaningful time series
        df['Month'] = df['Date'].dt.to_period('M')
        monthly_summary = df.groupby(['Month', 'Type'])['Amount'].sum().reset_index()
        monthly_summary['Month'] = monthly_summary['Month'].astype(str)
        
        fig_line = px.line(
            monthly_summary,
            x='Month',
            y='Amount',
            color='Type',
            title='Monthly Income vs Expenses Trend',
            markers=True,
            color_discrete_map={'Income': '#2e7d32', 'Expense': '#d32f2f'}
        )
        st.plotly_chart(fig_line, use_container_width=True)
        
        # Calculate savings rate by month
        pivot_monthly = monthly_summary.pivot(index='Month', columns='Type', values='Amount').fillna(0)
        if 'Income' in pivot_monthly.columns and 'Expense' in pivot_monthly.columns:
            pivot_monthly['Savings_Rate'] = ((pivot_monthly['Income'] - pivot_monthly['Expense']) / pivot_monthly['Income'] * 100).fillna(0)
            
            fig_savings = px.bar(
                x=pivot_monthly.index,
                y=pivot_monthly['Savings_Rate'],
                title='Monthly Savings Rate (%)',
                labels={'x': 'Month', 'y': 'Savings Rate (%)'},
                color=pivot_monthly['Savings_Rate'],
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_savings, use_container_width=True)
    
    # 3. Insights and recommendations
    st.subheader("💡 Financial Insights")
    
    insights = []
    
    if not expense_df.empty:
        top_category = expense_df.groupby('Category')['Amount'].sum().idxmax()
        top_amount = expense_df.groupby('Category')['Amount'].sum().max()
        insights.append(f"🎯 Your highest spending category is **{top_category}** at ${top_amount:,.2f}")
        
        avg_transaction = expense_df['Amount'].mean()
        insights.append(f"💰 Your average transaction amount is ${avg_transaction:.2f}")
        
        total_expenses = expense_df['Amount'].sum()
        total_income = df[df['Type'] == 'Income']['Amount'].sum()
        if total_income > 0:
            expense_ratio = (total_expenses / total_income) * 100
            insights.append(f"📊 You spend {expense_ratio:.1f}% of your income")
    
    for insight in insights:
        st.markdown(f"- {insight}")


def render_settings():
    """Render settings and data management"""
    st.title("⚙️ Settings & Data Management")
    
    st.subheader("💾 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Export Data**")
        if st.button("📁 Download Transactions (CSV)", use_container_width=True):
            df = tracker.get_transactions_df()
            if not df.empty:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="💾 Download CSV File",
                    data=csv,
                    file_name=f"transactions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No data to export")
    
    with col2:
        st.markdown("**Data Statistics**")
        metrics = tracker.calculate_metrics()
        st.metric("Total Transactions", metrics['transaction_count'])
        if st.session_state.get('last_saved'):
            st.write(f"Last saved: {st.session_state.last_saved.strftime('%Y-%m-%d %H:%M:%S')}")
    
    st.markdown("---")
    st.subheader("🗑️ Danger Zone")
    
    st.warning("⚠️ The following actions are irreversible!")
    
    if st.button("🗑️ Clear All Data", type="secondary"):
        if st.checkbox("I understand this will delete all my transactions"):
            st.session_state.transactions = []
            tracker.save_transactions()
            st.success("All data has been cleared!")
            st.rerun()


# Initialize the finance tracker
@st.cache_resource
def get_tracker():
    return StreamlitFinanceTracker()

tracker = get_tracker()


def main():
    """Main application entry point"""
    try:
        # Render sidebar and get current page
        current_page = render_sidebar()
        
        # Render the appropriate page
        if current_page == "dashboard":
            render_dashboard()
        elif current_page == "add_transaction":
            render_add_transaction()
        elif current_page == "view_transactions":
            render_view_transactions()
        elif current_page == "analytics":
            render_analytics()
        elif current_page == "settings":
            render_settings()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666; padding: 1rem;'>"
            "💰 Personal Finance Tracker | Built with Streamlit"
            "</div>",
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Please refresh the page or contact support if the problem persists.")


if __name__ == "__main__":
    main()
