# -*- coding: utf-8 -*-
"""
Mapmyrun Data Analysis Web App

The web app explores a user's mapmyrun data using 
python, streamlit, pandas and altair

To run:
    $streamlit run mapmyrun_app.py

History
v0.1.0 - Apr 2021, Initial version
v0.2.0 - Jan 2021, Updated to latest version of libraries
"""

import base64
import pandas as pd
import streamlit as st
import altair as alt

__author__ = "Terry Dolan"
__copyright__ = "Terry Dolan"
__license__ = "MIT"
__email__ = "terry8dolan@gmail.com"
__status__ = "Beta"
__version__ = "0.2.0"
__updated__ = "January 2022"

# explicitly register matplotlib converters to avoid warning
# Ref: https://stackoverflow.com/questions/47404653/pandas-0-21-0-timestamp-compatibility-issue-with-matplotlib
pd.plotting.register_matplotlib_converters()

# set streamlit page properties
st.set_page_config(page_title="mapmyrun_app",
                   layout="wide",
                   page_icon='redglider.ico',
                   initial_sidebar_state="collapsed")

st.title("Mapmyrun Data Analysis App")
with st.expander("Introduction", expanded=True):
    st.markdown("""
Use the app to explore your *[mapmyrun](https://www.mapmyrun.com/)* run data. 
**Configure the app** using the sidebar: load your own data and select the time 
period and view to plot. 
The plots are interactive.

Reveal the  sidebar by clicking the '**>**' in the top left of the app window.
Hide this *Introduction* section using the '**—**' to the far right of the
section name.
""")

# ==============================================
#    Mapmyrun Application - UTILITY FUNCTIONS
# ==============================================

def mmr_distance_cat(distance):
    """Return the distance category in km for the given distance."""

    if distance < 5:
        dist_cat = 'Lessthan5km'
    elif 5 <= distance < 10:
        dist_cat = '5to10km'
    elif distance >= 10:
        dist_cat = 'Morethan10km'

    return dist_cat

@st.cache
def mmr_read_data(file):
    """Read mapmyrun csv file and return running data in dataframe.

        Args:
            file: str, string with mapmyrun csv filename
            e.g. 'TD_workout_history_2021-01-31.csv'

        Returns:
            df: pd.DataFrame, pandas dataframe
            is_units_converted: Bool, True if miles converted to km

        Raises:
            Exception: if file cannot be parsed
    """

    # define multiplier to convert miles to kilometres
    MI_TO_KM = 1.60934

    try:
        print(f"\nreading mapmyrun data file: '{file}'")
        df = pd.read_csv(file, parse_dates=['Date Submitted',
                                            'Workout Date'])\
                                              .sort_values('Workout Date')\
                                              .reset_index(drop=True)

        # determine distance units from column names
        # will be either 'Distance (km)' or 'Distance (mi)'
        col_dist = [col for col in df.columns if col.startswith('Distance')]
        dist_units = col_dist[0][10:12] # km or mi

        # rename columns to remove whitespace and units (etc)
        col_map = {'Workout Date': 'Workout_date',
                   'Activity Type': 'Activity_type',
                   'Distance (km)': 'Distance',
                   'Distance (mi)': 'Distance',
                   'Workout Time (seconds)': 'Workout_time',
                   'Avg Pace (min/km)': 'Pace_avg',
                   'Avg Pace (min/mi)': 'Pace_avg',
                   'Avg Speed (km/h)': 'Speed_avg',
                   'Avg Speed (mi/h)': 'Speed_avg'}
        df = df.rename(columns=col_map)

        # drop unwanted columns
        col_keep = col_map.values()
        df = df.drop(columns=[col for col in df.columns
                              if col not in col_keep])

        # convert distance related values in dataframe to km
        # if distance units are mi
        is_units_converted = False
        if dist_units == 'mi':
            df['Distance'] = df.Distance*MI_TO_KM
            df['Pace_avg'] = df.Pace_avg/MI_TO_KM
            df['Speed_avg'] = df.Speed_avg*MI_TO_KM
            is_units_converted = True
            print('Distances in file converted from miles to kilometres')
    except:
        error_msg = f"error parsing the file: '{file.name}', "\
                    f"check that this is a mapmyrun csv file"
        print(error_msg)
        raise Exception(error_msg)

    # remove activity types that are not runs
    df = df[(df.Activity_type == 'Run')].drop(columns='Activity_type')

    # drop rows with zero distance and average speed of zero or speed > 30
    # these are assumed to be incomplete records
    df = df[(df.Distance != 0) & (df.Speed_avg != 0) & (df.Speed_avg <= 30)]

    # enrich data, add column with distance category
    df['Distance_category'] = df.Distance.apply(mmr_distance_cat)

    return df, is_units_converted

@st.cache
def mmr_lts(df):
    """Return lifetime stats (lts) in dict for given mmr dataframe."""

    print(f"Producing lifetime stats for dataframe")

    lts_d = dict()
    lts_d['total_runs'] = df.Workout_date.count()
    lts_d['total_dist'] = round(df.Distance.sum(), 1)
    lts_d['pace_avg'] = round(df.Pace_avg.mean(), 1)
    lts_d['speed_avg'] = round(df.Speed_avg.mean(), 1)

    # calculate max distance per month stats
    df_dist_pm = df[['Workout_date', 'Distance']]\
                        .groupby(pd.Grouper(key='Workout_date',
                                            freq='M')).sum()
    df_dist_pm = df_dist_pm.sort_values('Distance', ascending=False)
    lts_d['mmr_run_dist_pm_max'] = round(df_dist_pm.iloc[0].values[0], 1)
    lts_d['mmr_run_dist_pm_max_date'] = df_dist_pm.index[0].strftime("%b %Y")

    # calculate max distance per week stats
    df_dist_pw = df[['Workout_date', 'Distance']]\
                        .groupby(pd.Grouper(key='Workout_date',
                                            freq='W')).sum()
    df_dist_pw = df_dist_pw.sort_values('Distance', ascending=False)
    lts_d['mmr_run_dist_pw_max'] = round(df_dist_pw.iloc[0].values[0], 1)
    lts_d['mmr_run_dist_pw_max_date'] = df_dist_pw.index[0].strftime("%b %Y")

    # calculate 5 km and 10 km stats
    lts_d['total_5km'] = df[df.Distance >= 5].Workout_date.count()
    lts_d['total_10km'] = df[df.Distance >= 10].Workout_date.count()

    # calculate first and last 5 km date
    if lts_d['total_5km'] >= 1:
        lts_d['first_5km_date'] = df[df.Distance >= 5]\
                                      .Workout_date.iloc[0]\
                                      .strftime("%d %b %Y")
        lts_d['last_5km_date'] = df[df.Distance >= 5]\
                                     .Workout_date.\
                                     iloc[-1].strftime("%d %b %Y")
    else:
        lts_d['first_5km_date'] = None
        lts_d['last_5km_date'] = None

    # calculate first 10k date
    if lts_d['total_10km'] >= 1:
        lts_d['first_10km_date'] = df[df.Distance >= 10]\
                                       .Workout_date.iloc[0]\
                                       .strftime("%d %b %Y")
        lts_d['last_10km_date'] = df[df.Distance >= 10]\
                                       .Workout_date.iloc[-1]\
                                       .strftime("%d %b %Y")
    else:
        lts_d['first_10km_date'] = None
        lts_d['last_10km_date'] = None

    return lts_d

@st.cache
def mmr_df_filter(df, years, start_date, end_date):
    """Return dataframe filtered by given years, start_date and end_date.

       df is mmr dataframe (mandatory)
       years is a list of Workout_date years to include in plot (mandatory)
           e.g. years=[2020, 2021]
           - if no value is given (years=None) then all years in Workout_dates
             are included
       start_date and end_dates are dates to refine chosen years (mandatory)
           e.g. start_date='2020-01-01' (date string format is '%Y-%m-%d')
           - if no value is given then min and max Workout_dates are used,
             after years filter is applied

       note: years, start_date and end_date are not defaulted to None to
             ensure that streamlit caching will work"""

    print(f"Filter dataframe with years={years}, start_date={start_date}, "
          f"end_date={end_date}")

    # filter on years
    if years:
        df = df[df.Workout_date.dt.year.isin(years)]

    # set start_date and end_date if not input
    if not start_date:
        start_date = df[(df.Distance > 0)].Workout_date.min().date()\
                                            .strftime('%Y-%m-%d')
    if not end_date:
        end_date = df[(df.Distance > 0)].Workout_date.max().date()\
                                            .strftime('%Y-%m-%d')

    # filter on start and end dates
    df = df[df.Workout_date.between(pd.Timestamp(start_date),
                                    pd.Timestamp(end_date))]

    return df

def mmr_plot_dist(df, view='Month'):
    """Return the altair plot of Distance vs Workout_date for df and view.

       df is pandas mmr dataframe (mandatory)
       view is a string (optional), one of: 'Year', 'Month', 'Week', 'Day'
       - e.g. view='Day', default is 'Month'
    """

    print(f"Plot distance with view={view}")

    # defaults
    fig_width = 720
    fig_height = 300
    bar_col_default = 'steelblue'
    bar_col_highlight = 'darkred'
    sel_cols = ['Workout_date', 'Distance', 'Speed_avg']

    # define pandas grouper frequency map and aggregate functions map
    pd_freq_map = {'Year': 'YS', 'Month': 'MS', 'Week': 'W-MON', 'Day': 'D'}
    pd_agg_funcs_map = {'Distance': 'sum', 'Speed_avg': 'mean'}

    # define altair map, used to tailor Altair plot for each view
    alt_map = {'Month': {'x_shorthand':'Workout_date', 'x_title':'Month',
                         'tooltip_title': 'Run month',
                         'tooltip_date_format': '%b %Y',
                         'prop_title': 'Distance per month'
                        },
               'Year': {'x_shorthand':'Workout_date:O', 'x_title':'Year',
                        'tooltip_title': 'Run year',
                        'tooltip_date_format': '',
                        'prop_title': 'Distance per year'
                       },
               'Week': {'x_shorthand':'Workout_date', 'x_title':'Week Number',
                        'tooltip_title': 'Run week number',
                        'tooltip_date_format': '%V, %d %b %Y',
                        'prop_title': 'Distance per week'
                       },
               'Day': {'x_shorthand':'Workout_date', 'x_title':'Day',
                       'tooltip_title': 'Run date',
                       'tooltip_date_format': '%d %b %Y',
                       'prop_title': 'Distance per day'
                      }
              }

    # define view to plot using input view and start and end dates
    df_plot = df[sel_cols].groupby(pd.Grouper(key='Workout_date',
                                              freq=pd_freq_map[view]))\
                                        [['Distance', 'Speed_avg']]\
                                        .agg(pd_agg_funcs_map).reset_index()

    # prepare dataframe for plotting
    df_plot = df_plot.fillna(0)
    if view == 'Year':
        df_plot['Workout_date'] = df_plot.Workout_date.dt.year

    # plot the figure with Altair
    hover = alt.selection(type="single", empty='none', on='mouseover')

    fig = alt.Chart(df_plot).mark_bar(opacity=0.6).encode(
                x=alt.X(shorthand=alt_map[view]['x_shorthand'],
                        axis=alt.Axis(title=alt_map[view]['x_title'])),
                y=alt.Y('Distance', axis=alt.Axis(title='Distance (km)')),
                color=alt.condition(hover, alt.value(bar_col_highlight),
                                    alt.value(bar_col_default)),
                tooltip=[alt.Tooltip(shorthand='Workout_date',
                                     title=alt_map[view]['tooltip_title'],
                                     format=alt_map[view]
                                     ['tooltip_date_format']),
                         alt.Tooltip(shorthand='Distance',
                                     title='Run distance in km',
                                     format='.2f'),
                         alt.Tooltip(shorthand='Speed_avg',
                                     title='Run average speed in km/h',
                                     format='.2f')]
                ).add_selection(
                    hover
                ).properties(
                    title=alt_map[view]['prop_title'], width=fig_width,
                    height=fig_height
                )

    return fig, df_plot

def mmr_plot_speed(df):
    """Return a plot of run speed categories for given mmr dataframe.

        Plot includes trend line
    """

    # set defaults
    fig_width = 550
    fig_height = 180
    sel_cols = ['Workout_date', 'Speed_avg', 'Distance', 'Distance_category']

    # define dict to map from distance categories to title for plotting
    distcats_title_map = {'Lessthan5km': 'Less than 5 km',
                          '5to10km': '5 km to 10 km',
                          'Morethan10km': 'More than 10 km'}

    # define dict to map distance categories to order for plotting
    distcats_order_map = {'Lessthan5km': 0, '5to10km': 1, 'Morethan10km': 2}

    # define dataframe to plot
    df_plot = df[sel_cols]

    # prep for plot
    # define dict to map from distance categories to order to plot
    distcats_order_map = {'Lessthan5km': 0, '5to10km': 1, 'Morethan10km': 2}
    altair_col_map = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3',
                      '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']
    distcats_col_map = {'Lessthan5km': altair_col_map[0],
                        '5to10km': altair_col_map[1],
                        'Morethan10km': altair_col_map[2]}

    # determine the distance categories that are in scope for this plot
    distcats_scope = df_plot.Distance_category.unique()

    # set order of distance categories in scope
    distcats_scope_sorted = sorted(distcats_scope,
                                   key=lambda x: distcats_order_map[x])

    chart = {}
    for distcat in distcats_scope_sorted:
        #print(f"generate chart for chart[{distcat}]")
        chart[distcat] = alt.Chart(
                df_plot[(df_plot.Distance_category == distcat)]
            ).mark_circle(
                color=distcats_col_map[distcat],
                opacity=0.7
            ).encode(
                x=alt.X('Workout_date',
                        axis=alt.Axis(title='Run date')),
                y=alt.Y('Speed_avg',
                        axis=alt.Axis(title='Average speed per run (km/h)'),
                        scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip(shorthand='Workout_date',
                                title='Run Date',
                                format='%d %b %Y'),
                    alt.Tooltip(shorthand='Distance_category',
                                title='Run Category', format=''),
                    alt.Tooltip(shorthand='Speed_avg',
                                title='Run Average Speed', format='.2f')],
            ).properties(
                title=distcats_title_map[distcat],
                width=fig_width/len(distcats_scope), height=fig_height
            )
        chart[distcat] = chart[distcat] + chart[distcat]\
                            .transform_regression('Workout_date',
                                                  'Speed_avg')\
                            .mark_line(color=distcats_col_map[distcat],
                                       opacity=0.4, strokeWidth=3)

    charts = [chart[dc] for dc in distcats_scope_sorted]
    chart_final = alt.hconcat(*charts).resolve_scale(y='shared').properties(
        title="Average speed over time for run categories, with trend line"
        )

    return chart_final, df_plot

def mmr_plot_pace_bin(df):
    """Return a plot of run pace bin categories for given mmr dataframe."""

    # set defaults
    fig_width = 700
    fig_height = 300
    bar_width = 30
    bar_opacity = 0.8
    bin_minstep = 0.2
    y_minstep = 1
    sel_cols = ['Workout_date', 'Distance', 'Pace_avg', 'Distance_category']

    # select columns of interest
    df_plot = df[sel_cols]

    # plot the figure with Altair
    chart = alt.Chart(df_plot
        ).mark_bar(
            width=bar_width,
            opacity=bar_opacity
        ).transform_bin(
            "binned_pace", field="Pace_avg",
            bin=alt.Bin(nice=True, minstep=bin_minstep)
        ).encode(
            x=alt.X("binned_pace:Q", title="Average Run Pace (binned)",
                    axis=alt.Axis(format=".2f")),
            y=alt.Y("count():Q", title="Count of Runs",
                    axis=alt.Axis(tickMinStep=y_minstep)),
            color=alt.Color(
                "Distance_category",
                sort=["Lessthan5km", "5to10km", "Morethan10km"],
                legend=alt.Legend(title="Distance category")),
            tooltip=[alt.Tooltip("Distance_category:O",
                                 title="Distance Category"),
                     alt.Tooltip("binned_pace:Q",
                                 title="Average Pace (binned)",
                                 format=".2f"),
                     alt.Tooltip("count()", title="Count of Runs")]
        ).properties(
            title="Count of Runs at Pace",
            width=fig_width, height=fig_height
        )

    return chart, df_plot

def mmr_csvhelp_str():
    """Return help on how to download a mapmymyrun .csv file.

       Help is a string in markdown format.
       Ref: https://support.mapmyfitness.com/hc/en-us/articles/200118594-Export-Workout-Data
    """

    csvhelp_str = """
- To export your workout history, click
[here](https://www.mapmyfitness.com/workout/export/csv),
or copy and paste https://www.mapmyfitness.com/workout/export/csv into your
browser.
- Once you have done so, log in to your mapmyrun account using your email
and password.
- After you log in, a .csv file will be downloaded to your computer's
Downloads folder.
On your mobile device, the file will be opened once downloaded.
You can then tap the share icon at the bottom and send it to yourself via
email or another app.

Source:
[https://support.mapmyfitness.com/hc/en-us/articles/200118594-Export-Workout-Data]
(https://support.mapmyfitness.com/hc/en-us/articles/200118594-Export-Workout-Data)
"""
    return csvhelp_str

@st.cache
def read_file_str(filename):
    """Read the file and return as a string."""

    with open(filename, "r") as f:
        file_str = f.read()
    return file_str

# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def get_download_link_df_to_csv(df, filename):
    """Returns a link allowing given pandas dataframe to be downloaded as csv.

    Args:
        df: pandas dataframe
        filename: str, name of file
                  downloaded file will be <filename>.csv

        Returns:
            url href for use in markdown, to allow file to be downloaded

    """
    csv = df.to_csv(index=False)
    # strings <-> bytes conversions
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv"'\
           f'>Download data as csv file</a>'

    return href

# =========================================================
#    Mapmyrun Application - Sidebar area for user config
# =========================================================

# define example names and files
RUN1_NAME = 'Runner 1'
RUN1_FILE = r'data/TD_workout_history_2021-01-31.csv'
RUN2_NAME = 'Runner 2'
RUN2_FILE = r'data/JD_workout_history_2021-01-31.csv'
RUN3_NAME = 'Runner 3'
RUN3_FILE = r'data/Jam_workout_history_2021-01-09.csv'

# select runner and generate the dataframe
# set radio buttons to display horizontally (interim solution)
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>',
         unsafe_allow_html=True)
st.sidebar.subheader("Configure the app")
sel_runner = st.sidebar.radio("Select runner:", ('Your Data',
                                                 RUN1_NAME, RUN2_NAME,
                                                 RUN3_NAME), index=1)
uploaded_file = None
if sel_runner == RUN1_NAME:
    MAPMYRUN_CSV = RUN1_FILE
elif sel_runner == RUN2_NAME:
    MAPMYRUN_CSV = RUN2_FILE
elif sel_runner == RUN3_NAME:
    MAPMYRUN_CSV = RUN3_FILE
elif sel_runner == 'Your Data':
    # load user's csv data file
    with st.expander("Load Your Data", expanded=True):
        st.subheader('Select your mapmyrun csv data file to load')
        uploaded_file = st.file_uploader("Choose a file", type='csv')
        if st.checkbox('Show how to download your mapmayrun workhout history \
                       as a .csv file',
                       value=False):
            st.markdown(mmr_csvhelp_str())
        if uploaded_file is not None:
            MAPMYRUN_CSV = uploaded_file
        else:
            st.stop()
        # create placeholders in expander, so that info and error
        # related to loaded file can be shown in this section
        file_info_slot = st.empty()
        file_error_slot = st.empty()

# load the csv into a dataframe
try:
    df_mmr, units_converted = mmr_read_data(MAPMYRUN_CSV)
    if units_converted:
        file_info_slot.info('Distances in file converted from miles'
                            ' to kilometres')
except Exception as e:
    file_error_slot.error(e)
    st.stop()

# select the years and associated workout start and end dates to plot
years_l = list(df_mmr.Workout_date.dt.year.unique())
sel_years = st.sidebar.multiselect(label="Select years to include in plots:",
                                   options=years_l, default=years_l)
if not sel_years:
    st.warning("Please select at least one year!")
    st.stop()

# determine first and last workout dates for selected years
workout_dates = df_mmr[df_mmr.Workout_date.dt.year.isin(sel_years)]\
                            .Workout_date
workout_dates_first = workout_dates.iloc[0]
workout_dates_last = workout_dates.iloc[-1]

# refine dates to plot (optional)
sel_dates = st.sidebar.date_input('Refine the selected years with a date '
                                  'range:',
                                  [workout_dates_first, workout_dates_last],
                                  min_value=workout_dates_first,
                                  max_value=workout_dates_last)
sel_start_date = sel_dates[0]
sel_end_date = sel_dates[1]

# define start_date anbd end_date friendly name
sel_start_date_fn = sel_dates[0].strftime('%d %b %Y')
sel_end_date_fn = sel_dates[1].strftime('%d %b %Y')

if len(sel_dates) != 2:
    st.warning("Please select a date range, with a different start and "
               "end date, for the selected years!")
    st.stop()

# apply user selected filters to the dataframe
sel_view = st.sidebar.radio("Choose the summary view for the distance plot:",
                            ('Year', 'Month', 'Week', 'Day'), index=1)

# filter the dataframe using user values
df_mmr = mmr_df_filter(df_mmr, years=sel_years,
                       start_date=sel_start_date, end_date=sel_end_date)

# ==================================================
#    Mapmyrun Application - Main area, with plots
# ==================================================

st.header(f"Run Analysis for {sel_runner}")
view_friendly = {'Year': 'yearly', 'Month': 'monthly',
                 'Week': 'weekly', 'Day': 'daily'}
st.markdown(f"Showing plots from *{sel_start_date_fn}* to "
            f"*{sel_end_date_fn}* with a *{view_friendly[sel_view]}* view...")

with st.expander("Plot Distance", expanded=True):
    # plot the mapmyrun workout distance from dataframe
    fig_dist, df_dist = mmr_plot_dist(df_mmr, view=sel_view)
    st.write(fig_dist)
    # give option of displaying and downloading dataframe
    dist_df_placeholder = st.empty()
    dist_dl_placeholder = st.empty()
    if st.checkbox('Show Distance data', value=False):
        dist_df_placeholder.dataframe(df_dist)
        dist_dl_placeholder.markdown(get_download_link_df_to_csv(df_dist,
                                                                 'distance'),
                                     unsafe_allow_html=True)

with st.expander("Plot Speed", expanded=True):
    # plot the mapmyrun workout speed from dataframe
    fig_speed, df_speed = mmr_plot_speed(df_mmr)
    st.write(fig_speed)
    # give option of displaying and downloading dataframe
    speed_df_placeholder = st.empty()
    speed_dl_placeholder = st.empty()
    if st.checkbox('Show Speed data', value=False):
        # select df_speed columns for display
        df_speed = df_speed[['Workout_date', 'Speed_avg', 'Distance_category']]
        speed_df_placeholder.dataframe(df_speed)
        speed_dl_placeholder.markdown(get_download_link_df_to_csv(df_speed,
                                                                  'speed'),
                                      unsafe_allow_html=True)

with st.expander("Plot Pace", expanded=True):
    # plot the mapmyrun workout pace from dataframe
    fig_pace, df_pace = mmr_plot_pace_bin(df_mmr)
    st.write(fig_pace)
    # give option of displaying and downloading dataframe
    pace_df_placeholder = st.empty()
    pace_dl_placeholder = st.empty()
    if st.checkbox('Show Plot data', value=False):
        pace_df_placeholder.dataframe(df_pace)
        pace_dl_placeholder.markdown(get_download_link_df_to_csv(df_pace,
                                                                 'pace'),
                                     unsafe_allow_html=True)

# ==========================================================
#    Mapmyrun Application - Main area, supplementary info
# ==========================================================

# show lifetime stats
with st.expander("Lifetime stats"):
    # load lifetime stats into lts dict
    lts = mmr_lts(df_mmr)

    # split screen into columns and show the stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write("Total runs:")
        st.write("Average pace:")
        st.write("Max dist per month:")
        st.write("Max dist per week:")
        st.write("Total 5 km runs:")
        if lts['first_5km_date']:
            st.write("First 5 km run:")
        if lts['first_10km_date']:
            st.write("First 10 km run:")
    with col2:
        st.write(f"{lts['total_runs']}")
        st.write(f"{lts['pace_avg']} min/km")
        st.write(f"{lts['mmr_run_dist_pm_max']} km")
        st.write(f"{lts['mmr_run_dist_pw_max']} km")
        st.write(f"{lts['total_5km']}")
        if lts['first_5km_date']:
            st.write(f"{lts['first_5km_date']}")
        if lts['first_10km_date']:
            st.write(f"{lts['first_10km_date']}")
    with col3:
        st.write("Total distance:")
        st.write("Average speed:")
        st.write("Month of max:")
        st.write("Week of max:")
        st.write("Total 10 km runs:")
        if lts['last_5km_date']:
            st.write("Last 5 km run:")
        if lts['last_10km_date']:
            st.write("Last 10 km run:")
    with col4:
        st.write(f"{lts['total_dist']} km")
        st.write(f"{lts['speed_avg']} km/h")
        st.write(f"{lts['mmr_run_dist_pm_max_date']}")
        st.write(f"{lts['mmr_run_dist_pw_max_date']}")
        st.write(f"{lts['total_10km']}")
        if lts['last_5km_date']:
            st.write(f"{lts['last_5km_date']}")
        if lts['last_10km_date']:
            st.write(f"{lts['last_10km_date']}")

# Show info about the app
with st.expander("About the app"):
    ABOUT_MD_STR = read_file_str('README.md')
    st.markdown(ABOUT_MD_STR)
