
# Interface Specifications

This document defines the interaction interfaces between the layers in the Smart Track Management architecture.

## Presentation <-> Business Logic
- Method: `submitRequest(requestData)`
- Data Format: JSON
- Description: Sends user request data to business logic for processing.

## Business Logic <-> Data Access
- Method: `getTrainSchedule(trainId)`
- Method: `updateTrackStatus(trackId, status)`
- Description: Pulls and updates necessary data from the database.

## Data Access <-> Data Layer
- SQL queries or ORM-based fetch/update methods.
- Examples:
  - `SELECT * FROM schedules WHERE train_id = ?`
  - `UPDATE tracks SET status = ? WHERE track_id = ?`
