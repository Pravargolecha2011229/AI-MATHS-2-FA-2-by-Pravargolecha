## AI-MATHS-2-FA-2-by-Pravargolecha

# Last-Mile Delivery Analytics Dashboard

**FA-2: Dashboarding and Deployment** | Mathematics for AI-II | CRS: Artificial Intelligence
**Developed by:** Pravar Golecha

## Project Overview

This project presents an interactive business intelligence dashboard built using **Streamlit**. It analyzes last-mile delivery performance data to help logistics managers identify delivery delays, optimize fleet utilization, and enhance operational efficiency.

The objective is to transform raw delivery data into actionable insights by visualizing how weather, traffic, vehicle type, agent performance, geography, and product category influence delivery times and delay rates.


## Key Features

### Core Visualizations

1. **Delay Analyzer**
   Displays average delivery time by weather and traffic conditions using dual bar charts with line overlays for late delivery percentages.

2. **Vehicle Performance Comparison**
   Compares delivery efficiency across vehicle types using color-coded bar charts to support fleet optimization.

3. **Agent Performance Insights**
   Interactive scatter plot of agent rating vs. delivery time, color-coded by age group, with a trendline for performance correlation.

4. **Geographic Performance Heatmap**
   Heatmap showing average delivery times across delivery areas and product categories to identify regional bottlenecks.

5. **Category Distribution Analysis**
   Box plots visualizing delivery time distributions by product category, highlighting categories with consistent delays.

### Interactive Filters

* Weather Condition
* Traffic Level
* Vehicle Type
* Delivery Area
* Product Category

All visualizations and KPIs update dynamically based on filter selections.

### Key Performance Indicators (KPIs)

* Average Delivery Time (minutes)
* Late Delivery Rate (percentage)
* Total Deliveries (filtered count)
* Average Agent Rating (scale 1–5)

KPIs include delta values to compare filtered data to overall averages.

### Additional Features

* Export filtered summaries as CSV files
* Optional visualizations for detailed exploration
* Data quality report with cleaning operations and summary statistics
* Consistent and minimalistic UI styling using custom CSS


## Live Demo

**Dashboard URL:** [Insert Streamlit Cloud URL]

**GitHub Repository:** [https://github.com/YOUR_USERNAME/lastmile-delivery-analytics-dashboard](https://github.com/YOUR_USERNAME/lastmile-delivery-analytics-dashboard)


## Project Structure

```
lastmile-delivery-analytics-dashboard/
│
├── app.py                    # Main Streamlit application (686 lines)
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── .gitignore                # Git ignore rules
│
└── data/
    └── Last mile Delivery Data.xlsx   # Raw dataset
```

---

## Installation and Setup

### Prerequisites

* Python 3.8 or higher
* pip package manager
* Git (optional)

### Local Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/YOUR_USERNAME/lastmile-delivery-analytics-dashboard.git
   cd lastmile-delivery-analytics-dashboard
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Add the dataset**
   Place `Last mile Delivery Data.xlsx` in the `data/` folder.

5. **Run the application**

   ```bash
   streamlit run app.py
   ```

6. **Open in browser**
   Visit `http://localhost:8501` to access the dashboard.


## Data Processing Workflow

1. **Data Loading**
   Reads Excel data with flexible column detection using the `openpyxl` engine.

2. **Data Cleaning**

   * Handles missing values appropriately
   * Standardizes column names and text formatting
   * Converts data types and validates structure
   * Logs each cleaning operation

3. **Feature Engineering**

   * Creates a “late delivery” flag using statistical thresholds
   * Groups agent ages into <25, 25–40, and 40+
   * Categorizes delivery speed (Very Fast, Fast, Average, Slow)

4. **Aggregation**
   Groups data by relevant dimensions (Traffic, Weather, Vehicle, Area, Category) to compute mean delivery times and delay percentages.

5. **Dynamic Filtering**
   User selections in the sidebar instantly update all visualizations and metrics.


## Business Questions Addressed

| Question                                          | Visualization      | Outcome                          |
| ------------------------------------------------- | ------------------ | -------------------------------- |
| How do weather and traffic affect delivery times? | Delay Analyzer     | Identify high-risk conditions    |
| Which vehicles perform best?                      | Vehicle Comparison | Optimize fleet allocation        |
| Are higher-rated agents faster?                   | Agent Scatter      | Inform training and staffing     |
| Which areas face most delays?                     | Geographic Heatmap | Guide operational improvements   |
| Which product categories underperform?            | Category Boxplot   | Refine handling and SLA planning |


## Technology Stack

| Category        | Tool            | Purpose                              |
| --------------- | --------------- | ------------------------------------ |
| Framework       | Streamlit       | Dashboarding and interactivity       |
| Data Processing | Pandas          | Data manipulation and aggregation    |
| Computation     | NumPy           | Statistical and numerical operations |
| Visualization   | Plotly          | Interactive data visualization       |
| File Handling   | OpenPyXL        | Excel import and management          |
| Deployment      | Streamlit Cloud | Hosting and sharing                  |
| Version Control | GitHub          | Source code management               |


## Troubleshooting

| Issue                           | Solution                                            |
| ------------------------------- | --------------------------------------------------- |
| `ModuleNotFoundError: openpyxl` | Run `pip install openpyxl`                          |
| Slow dashboard performance      | Wait for caching to complete                        |
| Filters not updating            | Ensure each filter has at least one selected option |
| Missing required columns        | Verify the dataset includes all required columns    |


## Future Enhancements

* Time-series analysis for delivery trend forecasting
* Predictive model for delivery time estimation
* Real-time data updates via API integration
* Executive summary export (PDF)
* Role-based access control for users
* Integration with logistics management systems


## Author

**Pravar Golecha**
IBCP Student | CRS: Artificial Intelligence 


## Acknowledgments

* Streamlit and Plotly for the visualization framework
* FA-1 Storyboard for conceptual guidance and feature planning


## References

* [Streamlit Documentation](https://docs.streamlit.io/)
* [Plotly Python Documentation](https://plotly.com/python/)
* [Pandas Documentation](https://pandas.pydata.org/)
* [FA-1 Storyboard and Planning](./docs/)


**Developed by Pravar Golecha | FA-2 Mathematics for AI-II | October 2025**
