---

# **Product Requirements Document: NFL Offensive Projections Modeler V1.0**

|  |  |
| :---- | :---- |
| **Document Owner:** | Gemini |
| **Date:** | August 4, 2025 |
| **Status:** | Version 1.0 \- Scoping & Definition |

## **1\. Introduction**

This document outlines the requirements for the "NFL Offensive Projections Modeler," a data science project aimed at creating a robust system for forecasting NFL game and player statistics. The core of the project is to build a hierarchical model that first projects team-level offensive output on a game-by-game basis and then distributes these projected totals among the individual offensive skill players (QB, RB, WR, TE). This tool will serve as a foundational learning project for applying machine learning techniques to sports analytics, with an initial focus on leveraging the nfl\_data\_py Python library as the sole data source.

## **2\. Goals and Objectives**

The primary goal is to build a functional and accurate projection system with a clear, interactive command-line interface.

* **Objective 1: Team-Level Projections:** Develop a machine learning model that, given a team and their opponent for a specific game, accurately projects team-level offensive statistics.  
  * *Key Stats:* Pass Attempts, Completions, Passing Yards, Passing TDs, Rush Attempts, Rushing Yards, Rushing TDs.  
* **Objective 2: Player-Share Projections:** Develop a machine learning model to estimate the percentage of a team's offensive workload that each active player will receive.  
  * *Key Shares:* Target Share, Rushing Attempt Share, Positional Touchdown Shares.  
* **Objective 3: Player-Level Projections:** Combine the outputs from the team and player-share models to generate comprehensive stat-line projections for all primary offensive skill players.  
* **Objective 4: User Interface:** Create a simple, intuitive command-line interface (CLI) that allows a user to generate projections for a specific year, team, and time frame (full season or single week).

## **3\. Scope**

### **3.1. In-Scope (V1.0)**

* **Projection Timeframe:** Regular season games only for a user-specified future year (e.g., 2025).  
* **Projection Granularity:** Projections can be generated for a single week or aggregated for the entire regular season.  
* **Positions:** Offensive skill positions: Quarterback (QB), Running Back (RB), Wide Receiver (WR), Tight End (TE).  
* **Statistics:**  
  * **Passing:** Attempts, Completions, Yards, Touchdowns  
  * **Rushing:** Attempts, Yards, Touchdowns  
  * **Receiving:** Targets, Receptions, Yards, Touchdowns  
* **Data Source:** All data for model training and feature generation will be exclusively from the nfl\_data\_py library.  
* **Interface:** Command-Line Interface (CLI).

### **3.2. Out-of-Scope (V1.0)**

* **Post-season (Playoff) Projections:** The model will not project for playoff games.  
* **Other Positions:** Projections for Defensive, Kicker, or Special Teams players will not be included.  
* **External Data:** Integration of third-party data sources such as weather forecasts, live betting odds, or news sentiment is reserved for future versions.  
* **Real-time Updates:** The model will not adjust projections based on news that breaks mid-week (e.g., a player's practice status changing). The projection will be based on the latest available roster and injury data at the time of execution.  
* **Graphical User Interface (GUI):** A web or desktop application is not part of V1.0.

## **4\. User Persona & Experience**

* **Persona:** A data scientist or technically-inclined football analyst who is comfortable with command-line tools and interested in quantitative sports analysis, fantasy football, or betting.  
* **UX Flow:** The user interaction will follow a guided, step-by-step process within the terminal.  
  1. **Start Application:** The user runs the main Python script.  
  2. **Prompt for Year:** The system asks, Please enter the year you would like to project (e.g., 2025):.  
  3. **Prompt for Team:** The system fetches a list of all NFL team abbreviations using nfl\_data\_py.import\_team\_desc() and displays them. It then prompts, Please select a team from the list above (e.g., 'CIN'):.  
  4. **Prompt for Timeframe:** The system asks, Project full REGULAR season? (yes/no):.  
     * **If yes:** The application proceeds to generate projections for all 17 regular season games for the selected team.  
     * **If no:**  
       1. The system uses nfl\_data\_py.import\_schedules() to retrieve the team's schedule for the selected year.  
       2. It displays a numbered list of weekly matchups (e.g., 1\. Week 1 vs. KC, 2\. Week 2 @ CLE, etc.).  
       3. It prompts, Please select a week to project by its number:.  
  5. **Generate & Display Projections:** The system runs the full modeling pipeline and prints the final player projections to the console in a clean, tabular format (e.g., a pandas DataFrame).

## **5\. Functional Requirements**

### **FR1: Data Ingestion & Preparation**

1. **Data Sources:** The system will utilize the following functions from nfl\_data\_py:  
   * import\_pbp\_data(): Primary source for historical game situations and outcomes. Data will be aggregated to the game-team level to create features and targets for the team projection model.  
   * import\_weekly\_data() & import\_snap\_counts(): Primary sources for historical player usage (targets, carries, snaps) to create features for the player-share model.  
   * import\_schedules(): To retrieve future schedules and identify opponents.  
   * import\_weekly\_rosters(): To identify the roster of active players for the projection year.  
   * import\_injuries(): To filter out players who are confirmed to be inactive for a given game.  
   * import\_team\_desc(): To provide a canonical list of team abbreviations for user selection.  
2. **Data Integrity (Edge Case):** The system **must prevent future data leakage**. When projecting for a game in Year X, the training data for all models must only consist of data from seasons prior to Year X. No game data from Year X itself can be used for feature generation or training.

### **FR2: Team-Level Projection Model**

1. **Purpose:** To project the total offensive output for a team in a given game.  
2. **Training Data:** Each row in the dataset will represent a single team's performance in a single past game (team-game).  
3. **Features:**  
   * Team's historical offensive performance (e.g., rolling averages of yards per game, touchdowns per game, pass/run ratio from previous seasons).  
   * Opponent's historical defensive performance (e.g., rolling averages of yards allowed per game, turnovers forced per game from previous seasons).  
   * Game context: home, away, or neutral.  
   * Team-level efficiency metrics derived from PBP data (e.g., EPA per play, success rate).  
4. **Targets (Labels):** For each team-game, the actual stat totals:  
   * pass\_attempts, pass\_yards, pass\_tds, rush\_attempts, rush\_yards, rush\_tds.  
5. **Model:** A separate regression model (e.g., XGBoost, LightGBM, or Ridge) will be trained for each of the six target variables.

### **FR3: Player-Share Projection Model**

1. **Purpose:** To estimate the distribution of offensive opportunities among active players.  
2. **Active Player Identification:** For a given projection week, the system will use import\_weekly\_rosters() for the projection year to get the team roster and import\_injuries() to exclude any player with a game status of 'Out' or on 'Injured Reserve'.  
3. **Training Data:** Each row will represent a single player's performance in a single past game (player-game).  
4. **Features:**  
   * Player context: age, position, experience.  
   * Historical usage from previous seasons (e.g., rolling averages of target\_share, rush\_attempt\_share, snap\_percentage).  
   * Team context: number of active players at the same position.  
5. **Targets (Labels):** For each player-game, their calculated share of the offense:  
   * target\_share \= $\`\\frac{\\text{player\_targets}}{\\text{team\_pass\_attempts}}\`$  
   * rush\_attempt\_share \= $\`\\frac{\\text{player\_rush\_attempts}}{\\text{team\_rush\_attempts}}\`$  
   * (And similar calculations for red-zone opportunities to project touchdowns).  
6. **Model:** A gradient boosting regressor will be trained to predict these shares.  
7. **Constraint Adherence (Edge Case):** After predicting shares for all active players, the system must normalize them so that the sum of shares for any given category equals 100%. For example, if the raw predicted target shares sum to 1.1, each player's share will be divided by 1.1.  
   * ‘Sp′​=∑i=1n​Si​Sp​​‘, where Sp​ is the raw predicted share for player p.

### **FR4: Final Player Stat Calculation**

1. **Logic:** The system will deterministically combine the outputs from FR2 and FR3.  
2. **Formulas (Examples):**  
   * player\_proj\_targets \= team\_proj\_pass\_attempts × player\_proj\_target\_share  
   * player\_proj\_rush\_attempts \= team\_proj\_rush\_attempts × player\_proj\_rush\_attempt\_share  
   * Other stats like yards and receptions will be calculated by multiplying projected opportunities by a player's historical efficiency rates (e.g., catch\_rate, yards\_per\_target).

## **6\. Assumptions and Dependencies**

* **Assumption 1:** The nfl\_data\_py library will remain accessible and its data structure will be stable.  
* **Assumption 2:** Historical trends and player usage patterns are predictive of future performance.  
* **Assumption 3:** For future season projections, the most recently available roster is a sufficient proxy for the game-day roster. Player movement (trades/cuts) that occurs after the data is pulled is not accounted for.  
* **Dependency 1:** A stable internet connection is required to download data from the nflverse repositories.  
* **Dependency 2:** The user's machine must have a Python environment with necessary libraries (pandas, numpy, scikit-learn, xgboost, nfl\_data\_py) installed.

## **7\. Risks and Mitigations**

| Risk | Likelihood | Impact | Mitigation Strategy |
| :---- | :---- | :---- | :---- |
| **Model Inaccuracy** | Medium | High | Start with simple baseline models and establish robust cross-validation. Iteratively improve by feature engineering. Clearly document model performance metrics (MAE, RMSE) on a held-out test set. |
| **Rookie Projections** | High | Medium | V1 will not explicitly model rookies; they will have a zero or near-zero projection due to a lack of historical NFL data. This will be documented as a known limitation. V2 will address this using import\_draft\_picks and import\_combine\_data. |
| **Data Gaps** | Low | Medium | The nfl\_data\_py library is well-maintained. However, if certain data (e.g., snap counts for an old season) is missing, the system will gracefully handle it, possibly by excluding those years from the training set and logging a warning. |
| **Invalid User Input** | High | Low | Implement input validation at each prompt. For team selection, only allow abbreviations present in the provided list. For week selection, only allow numbers corresponding to the schedule. |

## **8\. Success Metrics**

* **Metric 1 (Model Performance):** The Mean Absolute Error (MAE) of projections against actuals, evaluated on a historical hold-out test set (e.g., the 2024 season). We will track MAE for key stats like passing yards and fantasy points.  
* **Metric 2 (Functionality):** The system successfully generates a projection for any valid team/year/week combination without runtime errors.  
* **Metric 3 (Usability):** The user can complete the UX flow from start to finish in under 2 minutes (excluding data download/model training time) and the final output is clear and understandable.

## **9\. Testing & Quality Assurance**

A comprehensive testing suite shall be developed concurrently with the application using a framework like `pytest`.

* **9.1. Unit Tests:** Each core function (e.g., feature calculation, data cleaning, share normalization) must have a corresponding unit test that verifies its correctness with known inputs and outputs.  
* **9.2. Data Validation Tests:** Tests will be created to assert the integrity of data at various stages of the pipeline. This includes checking for expected data types, ensuring no `NaN` values exist in critical columns, and programmatically verifying that no future data leakage occurs in the training set.  
* **9.3. Integration Tests:** The end-to-end pipeline, from data ingestion to final projection output, will be tested to ensure all components work together as expected. These tests can use cached, static data subsets to ensure speed and reproducibility.  
* **9.4. Model Regression Tests:** A fixed, held-out dataset (e.g., the complete 2024 season) will be used as a benchmark. After any significant change to the model or feature engineering process, the model's performance (e.g., MAE) on this benchmark set will be compared to the previous version. A significant decrease in performance will indicate a regression and fail the test.

## **10\. Future Work (Post V1.0)**

* **V1.1:** Integrate betting market data (`import_sc_lines`) as powerful features in the team projection model.  
* **V1.2:** Develop a dedicated model for rookie projections using college performance, draft capital (`import_draft_picks`), and combine results (`import_combine_data`).  
* **V1.3:** Incorporate external weather forecast data for games played outdoors.  
* **V2.0:** Expand projections to include defensive players (IDP) and kickers.  
* **V2.1:** Build a simple web-based user interface using Streamlit or Flask to make the tool more accessible.
