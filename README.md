# mmr # mmr [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]

Use the app to explore your *[mapmyrun](https://www.mapmyrun.com/)* run data. 

## Configure the app
Select one of the three example runner data files on the sidebar to see the 
plots and play with the data filters and views.
Or you can use *Your Data* file - follow the instructions on how to download
your .csv file from the mapmyrun website and to load the file into the app.

The app will show all years in the data set by default.
Use the *years* multi-select box in the sidebar to select (or deselect) the
years that you want to include in the plots.

The app will show all dates within the selected years by default.
Use the *date range* selector in the sidebar to define the start and end
dates that you want to include in the plots.

The distance plot will sum the distance data by the selected view; the 
default view is 'month'.
Use the *view* radio button in the sidebar to select an alternative view.

Note that the plots assume that the distance data is in km.
If your data has distance in miles then the distance related values are
automatically converted to km during the data load.

## Data viz and stats
The distance plot shows the runner's distance over time for the selected view.
Hovver your mouse pointer over a bar to see the details.

The speed plot shows the runner's average speed for each run over time, split
by pre-defined categories: less than 5 km, 5 km to 10 km, more than 10 km.
There is also a trend line showing if the speed is increasing or decreasing
over time for the selected time period. 
Hovver your mouse pointer over a point to see the details.

The pace plot shows the frequency of runs by pace. 
The bar is stacked showing the breakdown by run category.
Hovver your mouse pointer over a bar to see the details.

For each of the plots you can view the data and download as a .csv file.

The life-time stats section shows the runner's key stats e.g. total runs and 
total distance.

## Build information
The mmr app is built in *python* using *streamlit*, *altair*, *matplotlib* and
*base64*.

The app is deployed [here](https://share.streamlit.io/terrydolan/mmr/main/mapmyrun_app.py).  
Check out the source code repository [here](https://github.com/terrydolan/mmr).

## To Do
+ Allow user to choose the distance units (miles or kilometres) to display and
the speed categories in terms of those those units.
Currently the distance units are set to km and the speed categories are fixed
at: lessthan5km, 5to10km and morethan10km.

&nbsp;
&nbsp;

*Terry Dolan*  
*email: terry8dolan@gmail.com*  
*Twitter: @lfcsorted*