CREATE TABLE IF NOT EXISTS meters (
    meter_id SERIAL PRIMARY KEY,
    customer_name VARCHAR(100),
    installation_date DATE,
    status VARCHAR(20)
);

CREATE TABLE IF NOT EXISTS readings (
    reading_id SERIAL PRIMARY KEY,
    meter_id INT REFERENCES meters(meter_id),
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    usage_kwh DECIMAL(10, 4)
);

-- Insert Dummy Smart Meter Data
INSERT INTO meters (customer_name, installation_date, status) VALUES 
('Alice Smith', '2023-01-15', 'active'),
('Bob Jones', '2023-05-20', 'active'),
('Charlie Davis', '2024-02-10', 'inactive');

INSERT INTO readings (meter_id, usage_kwh) 
SELECT floor(random() * 3 + 1), random() * 5 FROM generate_series(1, 50);