import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Set page config
st.set_page_config(
    page_title="Option Chain Dashboard",
    page_icon="📊",
    layout="wide"
)

def generate_tv_level_text(df):
    lines = []

    def format_top(df, col, count=3, reverse=True, label=""):
        sorted_df = df.sort_values(col, ascending=not reverse).head(count)
        for i, row in enumerate(sorted_df.itertuples(), 1):
            lines.append(f"{label}{i}:{int(row.Strike)}")

    format_top(df, 'Absolute_GEX', label="AG")
    format_top(df[df.Net_GEX > 0], 'Net_GEX', label="P")
    format_top(df[df.Net_GEX < 0], 'Net_GEX', reverse=False, label="N")
    format_top(df, 'Cumulative_OI', label="HOI")
    format_top(df, 'Call_OpenInterest', label="COI")
    format_top(df, 'Put_OpenInterest', label="POI")

    return ";".join(lines)

def calculate_gex(df, lot_size):
    """Calculate Gamma Exposure (GEX) for options"""
    df = df.copy()

    # Calculate notional gamma for calls and puts
    df['Call_GEX'] = df['Call_Gamma'] * df['Call_OpenInterest'] * lot_size
    df['Put_GEX'] = df['Put_Gamma'] * df['Put_OpenInterest'] * lot_size * -1

    # Net GEX per strike
    df['Net_GEX'] = df['Call_GEX'] + df['Put_GEX']
    df['Absolute_GEX'] = abs(df['Call_GEX']) + abs(df['Put_GEX'])

    # GEX in lots
    df['Net_GEX_Lots'] = df['Net_GEX'] / lot_size
    df['Absolute_GEX_Lots'] = df['Absolute_GEX'] / lot_size

    return df

def find_gex_flip_point(df):
    """Find the strike where cumulative GEX flips from negative to positive"""
    df_sorted = df.sort_values('Strike')
    df_sorted['Cumulative_GEX'] = df_sorted['Net_GEX'].cumsum()

    # Find the flip point
    negative_to_positive = df_sorted[
        (df_sorted['Cumulative_GEX'].shift(1) < 0) &
        (df_sorted['Cumulative_GEX'] >= 0)
    ]

    if not negative_to_positive.empty:
        return negative_to_positive.iloc[0]['Strike']
    return None

def get_atm_range(df, range_strikes=20):
    """Get strikes within +/- range from ATM"""
    spot_price = df['Underlying_Price'].iloc[0]

    # Find ATM strike (closest to spot)
    df['Distance_from_Spot'] = abs(df['Strike'] - spot_price)
    atm_strike = df.loc[df['Distance_from_Spot'].idxmin(), 'Strike']

    # Get unique strikes and find strike interval
    strikes = sorted(df['Strike'].unique())
    if len(strikes) > 1:
        strike_interval = strikes[1] - strikes[0]
    else:
        strike_interval = 50  # default

    # Filter strikes within range
    min_strike = atm_strike - (range_strikes * strike_interval)
    max_strike = atm_strike + (range_strikes * strike_interval)

    return df[(df['Strike'] >= min_strike) & (df['Strike'] <= max_strike)]

def extract_underlying_name(underlying_string):
    """Extract underlying name from format like 'NSE:SPOT:NIFTY'"""
    try:
        return underlying_string.split(':')[-1]
    except:
        return underlying_string

def create_gex_chart(df, show_lots=True, lot_size=75):
    """Create combined GEX chart"""
    df_chart = get_atm_range(df)
    df_chart = df_chart.sort_values('Strike')

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Choose data based on toggle
    if show_lots:
        net_gex_data = df_chart['Net_GEX_Lots']
        abs_gex_data = df_chart['Absolute_GEX_Lots']
        y_title_primary = "Net GEX in Lots"
        y_title_secondary = "Absolute GEX in Lots"
    else:
        net_gex_data = df_chart['Net_GEX']
        abs_gex_data = df_chart['Absolute_GEX']
        y_title_primary = "Net GEX in Qty"
        y_title_secondary = "Absolute GEX in Qty"

    # Add Net GEX as bars (primary y-axis)
    fig.add_trace(
        go.Bar(
            x=df_chart['Strike'],
            y=net_gex_data,
            name='Net GEX',
            marker_color=['red' if x < 0 else 'green' for x in net_gex_data],
            opacity=0.7
        ),
        secondary_y=False,
    )

    # Add Absolute GEX as area chart (secondary y-axis)
    fig.add_trace(
        go.Scatter(
            x=df_chart['Strike'],
            y=abs_gex_data,
            fill='tonexty',
            mode='lines',
            name='Absolute GEX',
            line=dict(color='#bc2ac7', width=2, shape='spline'),
            fillcolor='rgba(188, 42, 199, 0.2)'
        ),
        secondary_y=True,
    )

    # Add spot price line
    spot_price = df['Underlying_Price'].iloc[0]
    fig.add_vline(
        x=spot_price,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Spot: {spot_price}"
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Strike Price", nticks=20, tickformat="d" )

    # Set y-axes titles
    fig.update_yaxes(title_text=y_title_primary, tickformat=".0f", secondary_y=False)
    fig.update_yaxes(title_text=y_title_secondary, tickformat=".0f", showgrid=False, secondary_y=True)

    # Update layout
    fig.update_layout(
        title="Gamma Exposure by Strike",
        hovermode='x unified',
        height=700,
        showlegend=True
    )

    return fig

def main():
    st.title("📊 Option Chain Dashboard")

    # File upload
    uploaded_file = st.file_uploader("Upload Option Chain CSV", type=['csv'])

    # Global toggles at the top
    col1, col2 = st.columns([5,1])
    with col1:
        show_lots = st.toggle("Show values in Lots", value=True)
    with col2:
        lot_size = st.number_input("Lot Size", min_value=1, value=75, step=1)
    unit = "Lots" if show_lots else "Qty"


    st.markdown("---")

    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)

            # Validate required columns
            required_columns = ['Strike', 'Underlying_Price', 'Call_OpenInterest', 'Put_OpenInterest', 'Call_Gamma', 'Put_Gamma']
            import io

            # Read raw file content to try detecting header row
            raw_text = uploaded_file.getvalue().decode("utf-8")
            for i, line in enumerate(raw_text.splitlines()[:10]):  # Check first 10 rows
                sample_df = pd.read_csv(io.StringIO(raw_text), skiprows=i)
                if all(col in sample_df.columns for col in required_columns):
                    df = sample_df
                    break
            else:
                st.error("Required columns not found in the first 10 rows. Please check the file format.")
                return

            # Calculate GEX
            df = calculate_gex(df, lot_size)

            # Display basic info
            st.subheader("📈 Market Overview")
            col1, col2, col3 = st.columns(3)

            with col1:
                underlying_full = df['Underlying'].iloc[0] if 'Underlying' in df.columns else "N/A"
                underlying = extract_underlying_name(underlying_full)
                st.metric("Underlying", underlying)

            with col2:
                spot_price = df['Underlying_Price'].iloc[0]
                st.metric("Spot Price", f"{spot_price:,.2f}")

            with col3:
                st.metric(f"Total Net GEX ({unit})", f"{(df['Net_GEX_Lots'].sum() if show_lots else df['Net_GEX'].sum()):,.0f}")

            st.markdown("---")

            # Top Statistics
            st.subheader("🏆 Top Statistics")

            # Total OI row
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"Total Call OI ({unit})", f"{(df['Call_OpenInterest'].sum()/lot_size if show_lots else df['Call_OpenInterest'].sum()):,.0f}")

            with col2:
                st.metric(f"Total Put OI ({unit})", f"{(df['Put_OpenInterest'].sum()/lot_size if show_lots else df['Put_OpenInterest'].sum()):,.0f}")

            with col3:
                flip_point = find_gex_flip_point(df)
                if flip_point:
                    st.metric("Gamma Flip", flip_point)
                else:
                    st.metric("Gamma Flip", "-")

            st.markdown("### 📊 Strikes with Highest Open Interest")

            # OI Statistics row
            col1, col2, col3 = st.columns(3)

            # Cumulative OI
            df['Cumulative_OI'] = df['Call_OpenInterest'] + df['Put_OpenInterest']

            with col1:
                unit = "Lots" if show_lots else "Qty"
                st.markdown(f"##### 🔵 Highest Cumulative OI ({unit})")
                top_cum_oi = df.nlargest(3, 'Cumulative_OI')[['Strike', 'Cumulative_OI']]
                for i, row in enumerate(top_cum_oi.itertuples(index=False, name="Row"), start=1):
                    value = row.Cumulative_OI / lot_size if show_lots else row.Cumulative_OI  # type: ignore
                    st.write(f"{i}. **{row.Strike}**: {value:,.0f}")  # type: ignore

            with col2:
                st.markdown(f"##### 🟢 Highest Call OI ({unit})")
                top_call_oi = df.nlargest(3, 'Call_OpenInterest')[['Strike', 'Call_OpenInterest']]
                for i, row in enumerate(top_call_oi.itertuples(index=False, name="Row"), start=1):
                    value = row.Call_OpenInterest / lot_size if show_lots else row.Call_OpenInterest  # type: ignore
                    st.write(f"{i}. **{row.Strike}**: {value:,.0f}")  # type: ignore

            with col3:
                st.markdown(f"##### 🔴 Highest Put OI ({unit})")
                top_put_oi = df.nlargest(3, 'Put_OpenInterest')[['Strike', 'Put_OpenInterest']]
                for i, row in enumerate(top_put_oi.itertuples(index=False, name="Row"), start=1):
                    value = row.Put_OpenInterest / lot_size if show_lots else row.Put_OpenInterest  # type: ignore
                    st.write(f"{i}. **{row.Strike}**: {value:,.0f}")  # type: ignore


            st.markdown("### 📉 Strikes with Highest Gamma Exposure")

            # GEX Statistics row
            col1, col2, col3 = st.columns(3)

            with col1:
                unit = "Lots" if show_lots else "₹"
                st.markdown(f"##### 🟣 Highest Absolute GEX ({unit})")
                top_abs_gex = df.nlargest(3, 'Absolute_GEX')[['Strike', 'Absolute_GEX', 'Absolute_GEX_Lots']]
                for i, row in enumerate(top_abs_gex.itertuples(index=False, name="Row"), start=1):
                    value = row.Absolute_GEX_Lots if show_lots else row.Absolute_GEX  # type: ignore

                    st.write(f"{i}. **{row.Strike}**: {value:,.0f}")  # type: ignore

            with col2:
                unit = "Lots" if show_lots else "₹"
                st.markdown(f"##### 🟢 Highest Positive GEX ({unit})")
                positive_gex = df[df['Net_GEX'] > 0]
                if not positive_gex.empty:
                    top_pos_gex = positive_gex.nlargest(3, 'Net_GEX')[['Strike', 'Net_GEX', 'Net_GEX_Lots']]
                    for i, row in enumerate(top_pos_gex.itertuples(index=False, name="Row"), start=1):
                        value = row.Net_GEX_Lots if show_lots else row.Net_GEX  # type: ignore

                        st.write(f"{i}. **{row.Strike}**: {value:,.0f}")  # type: ignore
                else:
                    st.write("-")

            with col3:
                unit = "Lots" if show_lots else "₹"
                st.markdown(f"##### 🔴 Highest Negative GEX ({unit})")
                negative_gex = df[df['Net_GEX'] < 0]
                if not negative_gex.empty:
                    top_neg_gex = negative_gex.nsmallest(3, 'Net_GEX')[['Strike', 'Net_GEX', 'Net_GEX_Lots']]
                    for i, row in enumerate(top_neg_gex.itertuples(index=False, name="Row"), start=1):
                        value = row.Net_GEX_Lots if show_lots else row.Net_GEX  # type: ignore

                        st.write(f"{i}. **{row.Strike}**: {value:,.0f}")  # type: ignore
                else:
                    st.write("-")


            st.markdown("---")

            # Charts Section
            st.subheader("🔄 Gamma Exposure Graph")

            # Create and display combined chart (removed dropdown)
            fig = create_gex_chart(df, show_lots, lot_size)
            st.plotly_chart(fig, use_container_width=True)

            # Data table
            st.markdown("---")
            st.subheader("📋 Detailed Data")

            # Filter for display
            display_df = get_atm_range(df)

            atm_strike = df.loc[df['Distance_from_Spot'].idxmin(), 'Strike']
            atm_strike = str(atm_strike)

            if show_lots:
                display_columns = ['Strike', 'Call_OpenInterest', 'Put_OpenInterest',
                                   'Cumulative_OI', 'Net_GEX_Lots', 'Absolute_GEX_Lots']

                display_formatted_raw = display_df[display_columns].copy()  # keep raw for logic
                display_formatted = display_formatted_raw.copy()

                # Format columns for display
                for col in ['Call_OpenInterest', 'Put_OpenInterest', 'Cumulative_OI']:
                    display_formatted[col] = (
                        display_formatted_raw[col]
                        .fillna(0)                 # Replace NaN with 0
                        .replace([np.inf, -np.inf], 0)  # Replace inf with 0
                        .astype(int)
                        .map('{:,}'.format)
                    )
                display_formatted['Net_GEX_Lots'] = (
                    display_formatted_raw['Net_GEX_Lots']
                    .fillna(0)
                    .replace([np.inf, -np.inf], 0)
                    .astype(int)
                    .map('{:,}'.format)
                )
                display_formatted['Absolute_GEX_Lots'] = (
                    display_formatted_raw['Absolute_GEX_Lots']
                    .fillna(0)
                    .replace([np.inf, -np.inf], 0)
                    .astype(int)
                    .map('{:,}'.format)
                )

                # Rename for display only
                display_formatted.columns = ['Strike', 'Call OI (Lots)', 'Put OI (Lots)',
                                             'Total OI (Lots)', 'Net GEX (Lots)', 'Abs GEX (Lots)']

            else:
                display_formatted_raw = display_df[['Strike', 'Call_OpenInterest', 'Put_OpenInterest',
                                                    'Cumulative_OI', 'Net_GEX', 'Absolute_GEX']].copy()
                display_formatted = display_formatted_raw.copy()

                for col in ['Call_OpenInterest', 'Put_OpenInterest', 'Cumulative_OI', 'Net_GEX', 'Absolute_GEX']:
                    display_formatted[col] = (
                        display_formatted_raw[col]
                        .fillna(0)                 # Replace NaN with 0
                        .replace([np.inf, -np.inf], 0)  # Replace inf with 0
                        .astype(int)
                        .map('{:,}'.format)
                    )


                display_formatted.columns = ['Strike', 'Call OI (Qty)', 'Put OI (Qty)',
                                             'Total OI (Qty)', 'Net GEX (₹)', 'Abs GEX (₹)']


            display_formatted_raw['Strike'] = display_formatted_raw['Strike'].astype(str)
            display_formatted['Strike'] = display_formatted['Strike'].astype(str)

            def highlight_dataframe(df):
                styles = pd.DataFrame('', index=df.index, columns=df.columns)

                # df_numeric = display_formatted_raw.set_index('Strike').loc[df['Strike']]
                df_numeric = display_formatted_raw.set_index('Strike').loc[df['Strike'].astype(str)]

                # Define mapping of display column name to raw column name and color
                highlight_rules = {
                    ('Call OI (Qty)' if not show_lots else 'Call OI (Lots)'): ('Call_OpenInterest', '#145A32'),       # green
                    ('Put OI (Qty)' if not show_lots else 'Put OI (Lots)'): ('Put_OpenInterest', '#78281F'),          # red
                    ('Total OI (Qty)' if not show_lots else 'Total OI (Lots)'): ('Cumulative_OI', '#154360'),         # blue
                    ('Abs GEX (₹)' if not show_lots else 'Abs GEX (Lots)'): ('Absolute_GEX' if not show_lots else 'Absolute_GEX_Lots', '#512E5F'),  # violet
                }


                net_gex_col_disp = 'Net GEX (₹)' if not show_lots else 'Net GEX (Lots)'
                net_gex_col_raw = 'Net_GEX' if not show_lots else 'Net_GEX_Lots'

                # Highlight top 3 for standard columns
                for col_disp, (col_raw, color) in highlight_rules.items():
                    if col_disp in df.columns:
                        top_3 = df_numeric[col_raw].nlargest(3).index
                        for i, strike in enumerate(df['Strike']):
                            if strike in top_3:
                                styles.iloc[i, df.columns.get_loc(col_disp)] = f'background-color: {color}; color: white'

                # Highlight top/bottom 3 in Net GEX
                top_3_pos = df_numeric[net_gex_col_raw].nlargest(3).index
                top_3_neg = df_numeric[net_gex_col_raw].nsmallest(3).index
                for i, strike in enumerate(df['Strike']):
                    if strike in top_3_pos:
                        styles.iloc[i, df.columns.get_loc(net_gex_col_disp)] = 'background-color: #145A32; color: white'  # green
                    elif strike in top_3_neg:
                        styles.iloc[i, df.columns.get_loc(net_gex_col_disp)] = 'background-color: #78281F; color: white'  # red

                # Highlight only the Strike column cell if it matches ATM
                for i, strike in enumerate(df['Strike']):
                    if strike == atm_strike:
                        styles.iloc[i, df.columns.get_loc('Strike')] = 'background-color: #b7950b; color: black; font-weight: bold'  # yellow

                return styles

            # Fix alignment of Strike column
            display_formatted['Strike'] = display_formatted['Strike'].astype(str)

            # Proceed with your styling
            styled_df = (
                display_formatted
                .style
                .apply(highlight_dataframe, axis=None)
            )
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                height=600
            )

            # === TradingView Auto-Level Section ===
            st.markdown("---")
            st.subheader("📌 TradingView Levels for Auto-Marking")
            tv_text = generate_tv_level_text(df)
            st.code(tv_text, language="text")


        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV has all the required columns.")

            # Show detailed error for debugging
            with st.expander("Debug Information"):
                st.text(f"Error details: {str(e)}")
                if 'df' in locals():
                    st.text(f"Available columns: {list(df.columns)}")

    else:
        st.info("👆 Please upload a CSV file to begin analysis")

        # Show expected format
        st.subheader("📝 Expected CSV Format")
        st.markdown("""
        Your CSV should contain the following key columns:
        - `Strike`: Strike prices
        - `Underlying_Price`: Current underlying price
        - `Underlying`: Underlying symbol (format: NSE:SPOT:NIFTY)
        - `Call_OpenInterest`: Open interest for calls
        - `Put_OpenInterest`: Open interest for puts
        - `Call_Gamma`: Gamma values for calls
        - `Put_Gamma`: Gamma values for puts
        - Plus other standard option chain columns...
        """)

if __name__ == "__main__":
    main()
