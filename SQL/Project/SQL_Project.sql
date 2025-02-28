create database sales;
use sales;

create table category(
category_id int primary key,
category_name varchar(25)
);

create table sellers(
seller_id int primary key,
seller_name varchar(25),
origin varchar(10)
);

create table customers(
customer_id int primary key,
first_name varchar(10),
last_name varchar(10),
state varchar(20)
);

create table inventory(
invetory_id int primary key,
product_id int,
stoke int,
warehouse_id int,
last_stoke_date date
);

create table order_items(
order_item_id int primary key,
order_id int,
product_id int,
quantity int,
prince_per_unit float
);

create table orders(
order_id int primary key,
order_date date,
customer_id int,
seller_id int,
order_status varchar(10)
);

create table payments(
payment_id int primary key,
order_id int,
payment_date date,
payment_status varchar(20)
);

create table products(
product_id int primary key,
product_name varchar(50),
price float,
cogs float,
category_id int
);

create table shipping(
shipping_id int primary key,
order_id int,
shipping_date date,
return_date date,
shipping_providers varchar(10),
delivery_status varchar(15)
);

load data infile 'C:\\Users\\panch\\OneDrive\\Desktop\\TOPPPS\\SQL\\Project\\datasets\\category.csv'
into table category
fields terminated by ','
enclosed by '"'
lines terminated by '\n'
ignore 1 rows;

load data infile 'C:\\Users\\panch\\OneDrive\\Desktop\\TOPPPS\\SQL\\Project\\datasets\\customers.csv'
into table customers
fields terminated by ','
enclosed by '"'
lines terminated by '\n'
ignore 1 rows;

load data infile 'C:\\Users\\panch\\OneDrive\\Desktop\\TOPPPS\\SQL\\Project\\datasets\\inventory.csv'
into table inventory
fields terminated by ','
enclosed by '"'
lines terminated by '\n'
ignore 1 rows;

load data infile 'C:\\Users\\panch\\OneDrive\\Desktop\\TOPPPS\\SQL\\Project\\datasets\\order_items.csv'
into table order_items
fields terminated by ','
enclosed by '"'
lines terminated by '\n'
ignore 1 rows;

load data infile 'C:\\Users\\panch\\OneDrive\\Desktop\\TOPPPS\\SQL\\Project\\datasets\\orders.csv'
into table orders
fields terminated by ','
enclosed by '"'
lines terminated by '\n'
ignore 1 rows;

load data infile 'C:\\Users\\panch\\OneDrive\\Desktop\\TOPPPS\\SQL\\Project\\datasets\\payments.csv'
into table payments
fields terminated by ','
enclosed by '"'
lines terminated by '\n'
ignore 1 rows;

load data infile 'C:\\Users\\panch\\OneDrive\\Desktop\\TOPPPS\\SQL\\Project\\datasets\\products.csv'
into table products
fields terminated by ','
enclosed by '"'
lines terminated by '\n'
ignore 1 rows;

load data infile 'C:\\Users\\panch\\OneDrive\\Desktop\\TOPPPS\\SQL\\Project\\datasets\\sellers.csv'
into table sellers
fields terminated by ','
enclosed by '"'
lines terminated by '\n'
ignore 1 rows;

LOAD DATA INFILE 'C:\\Users\\panch\\OneDrive\\Desktop\\TOPPPS\\SQL\\Project\\datasets\\shipping.csv'
INTO TABLE shipping
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(shipping_id, order_id, shipping_date, @return_date, shipping_providers, delivery_status)
SET return_date = NULLIF(@return_date, '');


alter table order_items add foreign key (order_id) references orders(order_id);
alter table order_items add foreign key (product_id) references products(product_id);
alter table products add foreign key (category_id) references category(category_id);
alter table orders add foreign key (customer_id) references customers(customer_id);
alter table orders add foreign key (seller_id) references sellers(seller_id);
alter table inventory add foreign key (product_id) references products(product_id);
alter table shipping add foreign key (order_id) references orders(order_id);
alter table payments add foreign key (order_id) references orders(order_id);


-- --------------------- Insights From Sales Dataset ---------------------------

-- Top 10 products by total sales value
-- Add total_sales Column
alter table order_items rename column prince_per_unit to price_per_unit;
alter table order_items add column total_sales float;
SET SQL_SAFE_UPDATES = 0;
update order_items set total_sales = quantity * price_per_unit;
SET SQL_SAFE_UPDATES = 1;

-- Answer
select p.product_id ,p.product_name, 
	sum(oi.total_sales) as total_sale,
	count(o.order_id) as total_orders
from products p 
join order_items oi on p.product_id = oi.product_id
join orders o on o.order_id = oi.order_id
group by p.product_id, p.product_name
order by total_sale desc limit 10;

-- --------------------- Insights ------------------------
-- Identifies which products generate the highest revenue.
-- Helps in understanding customer preferences and high-demand items.
-- Helps prioritize restocking and promotions.
-- Can track demand changes over time for better planning.



-- Total revenue generated by each product category
select c.category_id, c.category_name, sum(oi.total_sales) as total_revenue,
(sum(oi.total_sales)/(select sum(total_sales) from order_items) * 100) as contribution
from category c
join products p on c.category_id = p.category_id
join order_items oi on p.product_id = oi.product_id
group by c.category_id, c.category_name
order by total_revenue desc;

-- --------------------- Insights ------------------------
-- Identifies the most profitable product categories
-- Helps in understanding which categories drive business growth
-- Useful for inventory planning, marketing focus, and pricing adjustments



-- Find customers who have registered but never placed an order, listing customer details and the time since their registration.
select c.customer_id, 
    concat(c.first_name, ' ', c.last_name) as customer_name, 
    min(o.order_date) AS registration_date, 
    timestampdiff(month, min(o.order_date), curdate()) as months_since_registration
from customers c
left join orders o on c.customer_id = o.customer_id
where o.customer_id is null
group by c.customer_id, customer_name;

-- --------------------- Insights ------------------------
-- Identifies users who registered but never converted into buyers.
-- A high number of non-purchasing customers may indicate friction in the buying process.



-- Monthly total sales over the past year
with monthly_sale as (
select  concat(0,month(o.order_date), '-', year(o.order_date)) as month,
sum(oi.total_sales) as total_sales
from orders o
join order_items oi on o.order_id = oi.order_id
where o.order_date >= date_sub(curdate(), interval 12 month)
group by month)
select month, total_sales,
lag(total_sales, 1) over (order by month) as last_month_sales,
total_sales - lag(total_sales, 1) OVER (order by month) as sales_difference
from monthly_sale;

-- --------------------- Insights ------------------------
-- Identifies peak and slow months, helping to understand seasonal demand patterns.
-- sales_difference shows whether sales increased or declined compared to the previous month.



-- Best-selling product category for each state
with ranking as(
select c.state, ct.category_name, sum(oi.total_sales) as total_sales,
rank() over(partition by c.state order by sum(oi.total_sales) desc) as rankk
from orders o
join customers c on o.customer_id = c.customer_id
join order_items oi on o.order_id = oi. order_id
join products p on oi.product_id = p.product_id
join category ct on ct.category_id = p.category_id
group by c.state, ct.category_name)
select * from ranking 
where rankk = 1;

-- --------------------- Insights ------------------------
-- Helps understand which product categories dominate sales in different states.
-- Certain categories may perform better in specific states due to cultural, economic, or demographic factors.



-- Lifetime Spend or Purchase by Each Customer
select c.customer_id,
concat(c.first_name, ' ',  c.last_name) as full_name,
sum(total_sales) as Lifetime_spend
from orders o
join customers c on c.customer_id = o.customer_id
join order_items oi on oi.order_id = o.order_id
group by c.customer_id, full_name;

-- --------------------- Insights ------------------------
-- Helps classify customers into tiers (e.g., high, medium, low spenders) for personalized marketing.
-- Identifies top spenders who contribute the most revenue.



-- Shipping Delay (More than 7 days)
select o.order_id, o.order_date, s.shipping_id, s.shipping_date 
from orders o 
join customers c on c.customer_id = o.customer_id
join shipping s on o.order_id = s.order_id
where s.shipping_date - o.order_date > 7;

-- --------------------- Insights ------------------------
-- Helps detect frequent delays and assess supply chain efficiency
-- Delayed shipments can lead to dissatisfaction, cancellations, or negative reviews.
-- If delays are common, it may indicate issues with specific shipping providers.



-- Payment Success Rate
select p.payment_status, count(*) as total_count,
count(*)/(select count(*) from payments) * 100 as success_rate
from orders o
join payments p on o.order_id = p.order_id
group by p.payment_status;

-- --------------------- Insights ------------------------
-- A high failure rate may indicate issues with payment gateways, fraud detection systems, or customer errors.
-- Helps measure how often payments are successfully completed
-- Failed transactions can lead to frustration and potential loss of sales.



-- Top 10 Performing Sellers with Total Order
select s.seller_id, s.seller_name,
sum(oi.total_sales) as total_sales,
count(o.order_id) as total_orders 
from orders o
join sellers s on o.seller_id = s.seller_id
join order_items oi on oi.order_id = o.order_id
group by s.seller_id, s.seller_name
order by total_sales desc limit 10;

-- --------------------- Insights ------------------------
-- Highlights the most successful sellers driving business revenue.
-- Helps compare seller efficiency and contribution to overall revenue
-- Top sellers may deserve better commission rates, exclusive deals, or promotional boosts.



-- Product Profit Margin
select p.product_id, p.product_name,
sum(total_sales - (p.cogs * oi.quantity)) as profit_amount,
sum(total_sales - (p.cogs * oi.quantity)) / sum(total_sales) * 100 as profit_margin
from products p 
join order_items oi on p.product_id = oi.product_id
group by p.product_id, p.product_name
order by profit_margin desc;

-- --------------------- Insights ------------------------
-- Identifies products with the highest profit margins, not just high sales.
-- Low-margin products may need price adjustments or cost reductions.
-- Products with high sales but low profit margins might indicate high production or fulfillment costs.
-- High-margin products can be promoted more, while low-margin ones may require bundling or discontinuation.



-- Inactive Sellers
with last_six_month as
(select * from sellers
where seller_id not in(select seller_id from orders 
where order_date >= current_date - interval 6 month)
)
select o.seller_id,
max(o.order_date) as last_sale_date,
max(oi.total_sales) as last_sale_amount
from orders o
join 
last_six_month lsm on lsm.seller_id = o.seller_id
join order_items oi on o.order_id = oi.order_id
group by o.seller_id
order by last_sale_amount;



-- Top 5 Customers by Orders in Each State
select * from (
select c.state,
concat(c.first_name, ' ', c.last_name) as customers,
count(o.order_id) as total_orders,
sum(total_sales) as total_sale,
dense_rank() over(partition by c.state order by count(o.order_id) desc) as rankk
from orders o
join order_items oi on oi.order_id = o.order_id
join customers c on c.customer_id = o.customer_id
group by c.state, customers
) as t1 where rankk <=5;

-- --------------------- Insights ------------------------
-- Identifies sellers who may have stopped selling or disengaged from the platform.
-- If high-revenue sellers have become inactive, it could indicate business risks.
-- Inactivity may be due to competition, supply chain issues, or pricing inefficiencies.



-- The percentage of returning customers
with customer_orders as (
    select customer_id, count(order_id) as total_orders
    from orders group by customer_id
),
returning_customers as (
    select count(customer_id) as returning_count
    from customer_orders where total_orders > 1
),
total_customers as (
    select count(distinct customer_id) as total_count from orders)
select (rc.returning_count * 100.0 / tc.total_count) as returning_customer_percentage
from returning_customers rc
join total_customers tc on 1=1;

-- --------------------- Insights ------------------------
-- A 94% returning customer rate indicates high satisfaction and repeat purchases.
-- Your business likely has strong customer engagement, good service, and competitive pricing.
-- High retention suggests a strong brand presence and positive customer experience.



-- Churn Rate
with last_order as (
select customer_id, max(order_date) as last_order_date from orders
group by customer_id
),
churned_customers as (
select count(customer_id) as churned_customers from last_order
where last_order_date < date_sub(curdate(), interval 6 month)
),
total_customers as (
select count(distinct customer_id) as total_customers from customers
)
select(
cc.churned_customers * 100.0 / nullif(tc.total_customers, 0)) as churn_rate
from churned_customers cc
join total_customers tc on 1=1;

-- --------------------- Insights ------------------------
-- A 76% churn rate suggests that a large number of customers are not returning, which could indicate issues with customer satisfaction, pricing, or competition.
-- Despite a high returning customer rate (94%), a large portion of the customer base is still churning, meaning many customers buy once and never return.
-- If most of the active customers are repeat buyers, but overall churn is high, it might mean difficulty in attracting new long-term customers.
-- If churn is high despite strong sales, it may indicate issues like poor product quality, customer service gaps, or delivery delays.





