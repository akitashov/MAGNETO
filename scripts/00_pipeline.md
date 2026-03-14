```mermaid

flowchart TD
    %% --- STYLES ---
    classDef storage fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:black;
    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:black;
    classDef logic fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,stroke-dasharray: 5 5,color:black;
    classDef critical fill:#ffccbc,stroke:#bf360c,stroke-width:3px,color:black;
    classDef final fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px,color:black;

    %% --- 01. OMNI2 ETL ---
    subgraph S1 [01_Omni2_ETL.py: Space Weather]
        direction TB
        O_Raw[("OMNI Raw ZIP/DAT")]:::storage
        O_Parse["Parsing Line-by-Line<br/>(Handle 999.9 Fills)"]:::process
        O_Phys{{"Physics Calc:<br/>Ey = -V * Bz<br/>P = 1.67 * n * V^2"}}:::logic
        O_Limits["Physical Limits Filter<br/>(e.g. Speed 200-1500)"]:::process
        O_Agg["Resample to DAILY<br/>(Mean, Min, Max, Std)"]:::process
        O_Feat["Feature Gen:<br/>Moving Averages (5..450d)<br/>Storm Flags (Dst < -50)"]:::process
        O_Out[("omni_biosphere.feather")]:::storage
        
        O_Raw --> O_Parse --> O_Limits --> O_Phys --> O_Agg --> O_Feat --> O_Out
    end

    %% --- 02. MODIS ETL ---
    subgraph S2 [02_MODIS_ETL.py: Environment Env]
        direction TB
        M_Raw[("MODIS NetCDF (0.05°)")]:::storage
        M_Grid["Spatial Aggregation<br/>Target: 0.5° Grid (Mean)"]:::process
        M_ID{{"Gen IDs (Int16):<br/>lat_id = round(lat*100)<br/>lon_id = round(lon*100)"}}:::logic
        M_Time["Time Snapping<br/>(To 8-day Period Start)"]:::process
        M_Out[("MODIS_extract.parquet")]:::storage
        
        M_Raw --> M_Grid --> M_ID --> M_Time --> M_Out
    end

    %% --- 03. ERA5 ETL ---
    subgraph S3 [03_ERA5_t2m_ETL.py: Temperature Context]
        direction TB
        E_Raw[("ERA5 NetCDF (Hourly)")]:::storage
        E_Buffer{{"HOURLY BUFFER STRATEGY:<br/>Keep tail of previous file<br/>to bridge daily aggregations"}}:::critical
        E_Regrid["Regrid to 0.5°<br/>(Linear Interp)"]:::process
        E_Calc["K -> Celsius<br/>Hourly -> Daily Mean"]:::process
        E_Context["Context Gen:<br/>10-day Moving Avg"]:::process
        E_Out[("era5_temperature.parquet")]:::storage

        E_Raw --> E_Buffer --> E_Regrid --> E_Calc --> E_Context --> E_Out
    end

    %% --- 04. SIF ETL ---
    subgraph S4 [04_SIF_ETL.py: Biosphere SIF]
        direction TB
        S_Raw[("OCO-2 NetCDF (Daily)")]:::storage
        
        subgraph Modis_Prep [MODIS Preparation]
            S_LoadM["Load MODIS Year"]:::process
            S_Dilate{{"Spatial Dilation:<br/>Fill gaps using 8 neighbors"}}:::logic
        end

        subgraph SIF_Prep [SIF Raw Prep]
            S_Land["Filter: Land Only<br/>(Indicator == 0)"]:::process
            S_Snap["Snap to 0.5° Grid<br/>Gen Int16 IDs"]:::process
        end

        S_Merge["JOIN: SIF + MODIS<br/>Keys: Date, Lat_ID, Lon_ID"]:::critical
        S_Filter{{"ENV FILTERING:<br/>Cloud < 60%<br/>Aerosol < 50%<br/>LAI >= 0.4<br/>Quality Flags"}}:::logic
        S_AggYear["Agg Daily Means<br/>(Combine Swaths)"]:::process
        S_Out[("sif_aggregated.feather")]:::storage

        M_Out -.-> S_LoadM --> S_Dilate --> S_Merge
        S_Raw --> S_Land --> S_Snap --> S_Merge
        S_Merge --> S_Filter --> S_AggYear --> S_Out
    end

    %% --- 05. ANOMALY DETECTION ---
    subgraph S5 [05_SIF_anomalies.py: Modeling]
        direction TB
        A_Split["Group by Cell (Lat/Lon)<br/>Parallel Tasks"]:::process
        A_Model{{"Harmonic Regression:<br/>y = α + βt + γcos(ωt) + δsin(ωt)"}}:::logic
        A_Cache[("Disk Checkpoints<br/>(Batches)")]:::storage
        A_Resid["Calc Residuals:<br/>Observed - Predicted"]:::process
        A_Merge["Merge Checkpoints"]:::process
        A_Final[("sif_residuals.parquet")]:::storage

        S_Out --> A_Split --> A_Model --> A_Resid --> A_Cache --> A_Merge --> A_Final
    end

    %% --- 06. ANALYSIS ---
    subgraph S6 [06_Spearman.py: Statistical Analysis]
        direction TB
        
        subgraph Memory_Strat [Memory Optimization]
            An_Year{{"Year-by-Year Chunking:<br/>Load ERA5 only for current SIF year"}}:::critical
        end

        An_Join1["Inner Join: SIF Res + ERA5<br/>(Clean RAM after)"]:::process
        An_Join2["Join + OMNI Features"]:::process
        
        An_Bin{{"Temp Stratification:<br/>15 Equal-Size Bins"}}:::logic
        An_Corr["Spearman Correlation<br/>(per Bin, per Var)"]:::process
        
        An_Out_CSV[("Results.csv")]:::final
        An_Out_PQ[("Results.parquet")]:::final

        A_Final --> Memory_Strat
        E_Out --> Memory_Strat
        Memory_Strat --> An_Join1 --> An_Join2
        O_Out --> An_Join2
        An_Join2 --> An_Bin --> An_Corr --> An_Out_CSV & An_Out_PQ
    end```