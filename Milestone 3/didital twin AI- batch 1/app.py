# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
import os
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///digital_twin.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ------------------------- Context Processor -------------------------
@app.context_processor
def utility_processor():
    return dict(datetime=datetime)

# ------------------------- Gemini AI Setup -------------------------
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Install with: pip install google-generativeai")

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    print("Gemini initialized successfully.")
else:
    gemini_model = None
    if not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY not set. Assistant will use rule-based responses.")

# ------------------------- Database Models -------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    age = db.Column(db.Integer, default=25)
    occupation = db.Column(db.String(100), default='Student')
    monthly_income = db.Column(db.Float, default=50000)
    monthly_expenses = db.Column(db.Float, default=35000)
    savings = db.Column(db.Float, default=100000)
    study_hours_per_week = db.Column(db.Float, default=20)
    fitness_hours_per_week = db.Column(db.Float, default=5)
    sleep_hours_per_day = db.Column(db.Float, default=7)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)

    transactions = db.relationship('Transaction', backref='user', lazy=True)
    study_logs = db.relationship('StudyLog', backref='user', lazy=True)
    fitness_logs = db.relationship('FitnessLog', backref='user', lazy=True)
    goals = db.relationship('Goal', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Date, nullable=False, default=datetime.utcnow)
    category = db.Column(db.String(50))
    amount = db.Column(db.Float, nullable=False)
    type = db.Column(db.String(10))

class StudyLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Date, nullable=False, default=datetime.utcnow)
    hours = db.Column(db.Float, nullable=False)
    subject = db.Column(db.String(100))
    productivity_score = db.Column(db.Integer)

class FitnessLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Date, nullable=False, default=datetime.utcnow)
    activity = db.Column(db.String(50))
    duration_min = db.Column(db.Integer, nullable=False)
    calories_burned = db.Column(db.Integer)

class Goal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    description = db.Column(db.String(200))
    target_date = db.Column(db.Date)
    achieved = db.Column(db.Boolean, default=False)

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref='chat_messages')
# ------------------------- Helper Functions -------------------------
def generate_forecast(data, periods=6):
    if len(data) < 2:
        return [data[-1] if data else 0] * periods
    X = np.array(range(len(data))).reshape(-1, 1)
    y = np.array(data)
    model = LinearRegression()
    model.fit(X, y)
    future_X = np.array(range(len(data), len(data)+periods)).reshape(-1, 1)
    forecast = model.predict(future_X)
    noise = np.random.normal(0, np.std(y)*0.1 if np.std(y) > 0 else 1, periods)
    return (forecast + noise).tolist()

def calculate_financial_metrics(user):
    income = sum(t.amount for t in user.transactions if t.type == 'income')
    expenses = sum(t.amount for t in user.transactions if t.type == 'expense')
    net = income - expenses
    rate = (net/income*100) if income > 0 else 0
    return {'income': income, 'expenses': expenses, 'net': net, 'rate': rate}

def predict_gpa(user):
    logs = user.study_logs
    if not logs:
        return {'gpa': 3.0, 'efficiency': 70, 'recommendation': 'Log study sessions to get predictions.'}
    total_hours = sum(l.hours for l in logs)
    avg_prod = sum(l.productivity_score for l in logs) / len(logs)
    gpa = 2.0 + (total_hours/100) + (avg_prod/100)
    gpa = min(4.0, max(1.0, gpa))
    if total_hours < 20:
        rec = "Increase study hours."
    elif avg_prod < 70:
        rec = "Improve focus and quality."
    else:
        rec = "Keep up the good work!"
    return {'gpa': round(gpa, 2), 'efficiency': round(avg_prod, 1), 'recommendation': rec}

def simulate_scenario(user, action, amount):
    scenarios = {
        'save': {
            'desc': f'Save ₹{amount:,} monthly',
            'impact': f'Your savings grow to ₹{user.savings + amount*12:,.0f} in 1 year.',
            'risk': 'Low',
            'rec': 'Great for long-term security'
        },
        'invest': {
            'desc': f'Invest ₹{amount:,} monthly',
            'impact': f'With 10% returns, investment grows to ₹{amount*12*1.1:,.0f} in 1 year.',
            'risk': 'Medium',
            'rec': 'Balanced approach for growth'
        },
        'spend': {
            'desc': f'Spend ₹{amount:,} monthly extra',
            'impact': f'Your savings reduce to ₹{user.savings - amount*12:,.0f} in 1 year.',
            'risk': 'High',
            'rec': 'Consider reducing discretionary spending'
        },
        'study_more': {
            'desc': f'Study {amount} more hours per week',
            'impact': f'Your GPA could increase by {amount*0.1:.2f} points.',
            'risk': 'Low',
            'rec': 'Consistent effort yields results'
        },
        'exercise': {
            'desc': f'Exercise {amount} minutes daily',
            'impact': f'You could burn {amount*30*30:,.0f} calories monthly.',
            'risk': 'Low',
            'rec': 'Great for overall health'
        }
    }
    return scenarios.get(action, scenarios['save'])

# ------------------------- Default Admin Creator -------------------------
def create_default_admin():
    """Create a default admin user if none exists."""
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin = User(
            username='admin',
            email='admin@digitaltwin.ai'
        )
        admin.set_password('admin123')
        admin.is_admin = True
        db.session.add(admin)
        db.session.commit()
        print("✅ Default admin user created:")
        print("   Username: admin")
        print("   Password: admin123")
        print("   Please change your password after first login.")
    else:
        print("✅ Admin user already exists.")

# ------------------------- Routes -------------------------
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        if not username or not email or not password:
            flash('All fields are required.')
            return redirect(url_for('register'))
        if password != confirm_password:
            flash('Passwords do not match.')
            return redirect(url_for('register'))
        if len(password) < 8:
            flash('Password must be at least 8 characters.')
            return redirect(url_for('register'))
        if User.query.filter_by(username=username).first():
            flash('Username already exists.')
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash('Email already registered.')
            return redirect(url_for('register'))
        user = User(username=username, email=email)
        user.set_password(password)
        # First user becomes admin – but now we already have a default admin, so this is optional
        # if User.query.count() == 0:
        #     user.is_admin = True
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            flash('Please enter both username and password.')
            return render_template('login.html')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('admin_dashboard' if user.is_admin else 'user_dashboard'))
        flash('Invalid username or password.')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# ------------------------- User Routes (unchanged) -------------------------
@app.route('/user/dashboard')
@login_required
def user_dashboard():


    # ===== Existing metrics =====
    metrics = calculate_financial_metrics(current_user)
    gpa_info = predict_gpa(current_user)
    total_fitness = sum(f.duration_min for f in current_user.fitness_logs)

    # ===== Chart Data Preparation (as before) =====
    monthly_income = defaultdict(float)
    monthly_expenses = defaultdict(float)
    for tx in current_user.transactions:
        month_key = tx.date.strftime('%Y-%m')
        if tx.type == 'income':
            monthly_income[month_key] += tx.amount
        else:
            monthly_expenses[month_key] += tx.amount

    all_months = sorted(set(monthly_income.keys()) | set(monthly_expenses.keys()))
    months = all_months[-6:]
    income_data = [monthly_income.get(m, 0) for m in months]
    expense_data = [monthly_expenses.get(m, 0) for m in months]

    study_dates = defaultdict(float)
    for log in current_user.study_logs:
        study_dates[log.date.strftime('%Y-%m-%d')] += log.hours
    study_days = sorted(study_dates.keys())[-14:]
    study_hours = [study_dates.get(d, 0) for d in study_days]

    fitness_dates = defaultdict(float)
    for log in current_user.fitness_logs:
        fitness_dates[log.date.strftime('%Y-%m-%d')] += log.duration_min
    fitness_days = sorted(fitness_dates.keys())[-14:]
    fitness_minutes = [fitness_dates.get(d, 0) for d in fitness_days]

    goal_progress = []
    for goal in current_user.goals:
        if not goal.achieved and goal.target_date:
            days_remaining = (goal.target_date - datetime.now().date()).days
            if days_remaining < 0:
                progress = 100
            else:
                progress = min(100, max(0, 100 - (days_remaining / 30 * 100)))
            goal_progress.append({'description': goal.description, 'progress': round(progress, 1)})

    # ===== PREDICTIVE ANALYTICS =====
    # Savings forecast (same as before)
    monthly_net = defaultdict(float)
    for tx in current_user.transactions:
        month_key = tx.date.strftime('%Y-%m')
        if tx.type == 'income':
            monthly_net[month_key] += tx.amount
        else:
            monthly_net[month_key] -= tx.amount
    sorted_months = sorted(monthly_net.keys())
    net_values = [monthly_net[m] for m in sorted_months[-6:]]
    if len(net_values) >= 3:
        forecast_net = generate_forecast(net_values, 6)
        savings_forecast = []
        cum = current_user.savings
        for val in forecast_net:
            cum += val
            savings_forecast.append(cum)
    else:
        savings_forecast = [current_user.savings + i * 5000 for i in range(6)]

    # GPA forecast
    study_hours_by_month = defaultdict(float)
    for log in current_user.study_logs:
        month_key = log.date.strftime('%Y-%m')
        study_hours_by_month[month_key] += log.hours
    study_hours_list = [study_hours_by_month[m] for m in sorted(study_hours_by_month.keys())[-4:]]
    if len(study_hours_list) >= 3:
        forecast_hours = generate_forecast(study_hours_list, 3)
        gpa_forecast = []
        avg_prod = sum(l.productivity_score for l in current_user.study_logs) / len(current_user.study_logs) if current_user.study_logs else 70
        for hours in forecast_hours:
            gpa = 2.0 + (hours / 50) + (avg_prod / 200)
            gpa_forecast.append(round(min(4.0, max(1.0, gpa)), 2))
    else:
        gpa_forecast = [gpa_info['gpa'] + i*0.05 for i in range(3)]

    # Fitness projection – use the same logic as before
    fitness_by_day = defaultdict(float)
    for log in current_user.fitness_logs:
        fitness_by_day[log.date.strftime('%Y-%m-%d')] += log.duration_min
    fitness_days_list = list(fitness_by_day.values())[-14:]
    if len(fitness_days_list) >= 5:
        forecast_fitness = generate_forecast(fitness_days_list, 30)
        projected_fitness = sum(forecast_fitness)
    else:
        projected_fitness = total_fitness * 2

    # ===== EXTRA METRICS =====
    total_study_hours = sum(log.hours for log in current_user.study_logs) or 0

    # Study streak (consecutive days with at least some study in last 7 days)
    study_streak = 0
    if current_user.study_logs:
        study_days_set = set(log.date for log in current_user.study_logs)
        today = datetime.now().date()
        for i in range(7):
            if today - timedelta(days=i) in study_days_set:
                study_streak += 1
            else:
                break

    # Fitness streak (consecutive days with >= 10 min activity)
    fitness_streak = 0
    if current_user.fitness_logs:
        today = datetime.now().date()
        for i in range(7):
            day_total = sum(l.duration_min for l in current_user.fitness_logs if l.date == (today - timedelta(days=i)))
            if day_total >= 10:
                fitness_streak += 1
            else:
                break

    # Subject breakdown
    subject_totals = defaultdict(float)
    for log in current_user.study_logs:
        if log.subject:
            subject_totals[log.subject] += log.hours
    subjects = list(subject_totals.keys())
    subject_hours = list(subject_totals.values())

    # Activity breakdown
    activity_totals = defaultdict(float)
    for log in current_user.fitness_logs:
        activity_totals[log.activity] += log.duration_min
    activities = list(activity_totals.keys())
    activity_minutes = list(activity_totals.values())

    # Life Score
    life_score = (
        (metrics['rate'] / 30) * 20 +
        (gpa_info['gpa'] / 4) * 30 +
        (min(total_fitness, 300) / 300) * 25 +
        (len([g for g in current_user.goals if not g.achieved]) / max(1, len(current_user.goals))) * 25
    )
    life_score = round(life_score)

    # ===== RENDER =====
    return render_template('user/dashboard.html',
                           user=current_user,
                           metrics=metrics,
                           gpa=gpa_info,
                           total_fitness=total_fitness,
                           months=months,
                           income_data=income_data,
                           expense_data=expense_data,
                           study_days=study_days,
                           study_hours=study_hours,
                           fitness_days=fitness_days,
                           fitness_minutes=fitness_minutes,
                           goal_progress=goal_progress,
                           savings_forecast=savings_forecast,
                           gpa_forecast=gpa_forecast,
                           projected_fitness=projected_fitness,
                           total_study_hours=total_study_hours,
                           study_streak=study_streak,
                           fitness_streak=fitness_streak,
                           subjects=subjects,
                           subject_hours=subject_hours,
                           activities=activities,
                           activity_minutes=activity_minutes,
                           life_score=life_score)

@app.route('/user/financial', methods=['GET', 'POST'])
@login_required
def user_financial():


    if request.method == 'POST':
        date = datetime.strptime(request.form['date'], '%Y-%m-%d')
        category = request.form['category']
        amount = float(request.form['amount'])
        trans_type = request.form['type']
        tx = Transaction(user_id=current_user.id, date=date, category=category, amount=amount, type=trans_type)
        db.session.add(tx)
        db.session.commit()
        flash('Transaction added.')
        return redirect(url_for('user_financial'))

    transactions = current_user.transactions
    expense_data = [t.amount for t in transactions if t.type == 'expense']
    forecast = generate_forecast(expense_data, 6) if expense_data else []

    # Metrics
    metrics = calculate_financial_metrics(current_user)

    # Category breakdown for pie chart (only expenses)
    category_totals = {}
    for tx in transactions:
        if tx.type == 'expense':
            category_totals[tx.category] = category_totals.get(tx.category, 0) + tx.amount
    categories = list(category_totals.keys())
    category_amounts = list(category_totals.values())

    # For the recent transactions list, we'll limit to last 10
    recent_transactions = transactions[-10:][::-1]  # newest first

    return render_template('user/financial.html',
                           user=current_user,
                           transactions=recent_transactions,
                           forecast=forecast,
                           metrics=metrics,
                           categories=categories,
                           category_amounts=category_amounts)


@app.route('/user/study', methods=['GET', 'POST'])
@login_required
def user_study():


    if request.method == 'POST':
        date = datetime.strptime(request.form['date'], '%Y-%m-%d')
        hours = float(request.form['hours'])
        subject = request.form['subject']
        productivity = int(request.form['productivity'])
        log = StudyLog(user_id=current_user.id, date=date, hours=hours, subject=subject,
                       productivity_score=productivity)
        db.session.add(log)
        db.session.commit()
        flash('Study session logged.')
        return redirect(url_for('user_study'))

    logs = current_user.study_logs
    gpa_info = predict_gpa(current_user)

    # --- Data for charts ---
    # 1. Daily study hours (last 14 days)
    from collections import defaultdict
    study_by_day = defaultdict(float)
    for log in logs:
        study_by_day[log.date.strftime('%Y-%m-%d')] += log.hours
    study_dates = sorted(study_by_day.keys())[-14:]
    study_hours = [study_by_day.get(d, 0) for d in study_dates]

    # 2. Subject breakdown (total hours per subject)
    subject_totals = defaultdict(float)
    for log in logs:
        if log.subject:
            subject_totals[log.subject] += log.hours
    subjects = list(subject_totals.keys())
    subject_hours = list(subject_totals.values())

    # 3. Productivity trend (last 14 days) – we need average productivity per day
    prod_by_day = defaultdict(list)
    for log in logs:
        prod_by_day[log.date.strftime('%Y-%m-%d')].append(log.productivity_score)
    prod_dates = sorted(prod_by_day.keys())[-14:]
    prod_avg = [sum(prod_by_day[d])/len(prod_by_day[d]) if d in prod_by_day else 0 for d in prod_dates]

    # 4. Total metrics
    total_hours = sum(log.hours for log in logs)
    avg_productivity = sum(log.productivity_score for log in logs) / len(logs) if logs else 0
    # Study streak: consecutive days with at least some study in last 7 days
    study_streak = 0
    if logs:
        # Get all unique study days, sorted descending
        study_days_set = sorted(set(log.date for log in logs), reverse=True)
        # Check from today backwards
        today = datetime.now().date()
        for i in range(7):
            check_date = today - timedelta(days=i)
            if check_date in study_days_set:
                study_streak += 1
            else:
                break

    # Recent logs (last 10, newest first)
    recent_logs = logs[-10:][::-1]

    return render_template('user/study.html',
                           user=current_user,
                           logs=recent_logs,
                           gpa=gpa_info,
                           study_dates=study_dates,
                           study_hours=study_hours,
                           subjects=subjects,
                           subject_hours=subject_hours,
                           prod_dates=prod_dates,
                           prod_avg=prod_avg,
                           total_hours=total_hours,
                           avg_productivity=avg_productivity,
                           study_streak=study_streak)


@app.route('/user/fitness', methods=['GET', 'POST'])
@login_required
def user_fitness():


    if request.method == 'POST':
        date = datetime.strptime(request.form['date'], '%Y-%m-%d')
        activity = request.form['activity']
        duration = int(request.form['duration'])
        calories = int(request.form['calories'])
        log = FitnessLog(user_id=current_user.id, date=date, activity=activity,
                         duration_min=duration, calories_burned=calories)
        db.session.add(log)
        db.session.commit()
        flash('Fitness activity logged.')
        return redirect(url_for('user_fitness'))

    logs = current_user.fitness_logs

    # --- Basic metrics ---
    total_min = sum(l.duration_min for l in logs)
    total_cal = sum(l.calories_burned for l in logs)
    avg_min_per_day = total_min / max(1,
                                      (datetime.now().date() - (logs[0].date if logs else datetime.now().date())).days)

    # --- Weekly progress (current week: Monday–Sunday) ---
    today = datetime.now().date()
    start_of_week = today - timedelta(days=today.weekday())  # Monday
    weekly_logs = [l for l in logs if l.date >= start_of_week]
    weekly_min = sum(l.duration_min for l in weekly_logs)
    week_goal = 150  # minutes per week
    week_progress = min(100, (weekly_min / week_goal) * 100)

    # --- Daily trend (last 14 days) ---
    from collections import defaultdict
    daily_totals = defaultdict(int)
    for log in logs:
        daily_totals[log.date.strftime('%Y-%m-%d')] += log.duration_min
    # Get last 14 days (including today)
    dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(13, -1, -1)]
    daily_data = [daily_totals.get(d, 0) for d in dates]

    # --- Activity breakdown ---
    activity_totals = defaultdict(int)
    for log in logs:
        activity_totals[log.activity] += log.duration_min
    activities = list(activity_totals.keys())
    activity_minutes = list(activity_totals.values())

    # --- Streak: consecutive days with at least 10 min of activity ---
    streak = 0
    if logs:
        study_days_set = sorted(set(log.date for log in logs), reverse=True)
        for i in range(30):  # max 30 days streak
            check_date = today - timedelta(days=i)
            # Check if there is any activity on this day (>=10 min)
            day_total = sum(l.duration_min for l in logs if l.date == check_date)
            if day_total >= 10:
                streak += 1
            else:
                break

    # --- Recent logs (last 10, newest first) ---
    recent_logs = logs[-10:][::-1]

    return render_template('user/fitness.html',
                           user=current_user,
                           logs=recent_logs,
                           total_min=total_min,
                           total_cal=total_cal,
                           avg_min_per_day=round(avg_min_per_day, 1),
                           weekly_min=weekly_min,
                           week_progress=round(week_progress, 1),
                           dates=dates,
                           daily_data=daily_data,
                           activities=activities,
                           activity_minutes=activity_minutes,
                           streak=streak)

@app.route('/user/simulation', methods=['GET', 'POST'])
@login_required
def user_simulation():

    result = None
    current_values = {
        'savings': current_user.savings,
        'gpa': predict_gpa(current_user)['gpa'],
        'fitness': sum(f.duration_min for f in current_user.fitness_logs)
    }

    if request.method == 'POST':
        scenario = request.form['scenario']
        amount = float(request.form['amount'])
        result = simulate_scenario(current_user, scenario, amount)
        # Add a projected value for comparison
        if scenario == 'save':
            result['projected'] = current_user.savings + amount * 12
            result['current'] = current_user.savings
        elif scenario == 'invest':
            result['projected'] = amount * 12 * 1.1
            result['current'] = 0
        elif scenario == 'spend':
            result['projected'] = current_user.savings - amount * 12
            result['current'] = current_user.savings
        elif scenario == 'study_more':
            result['projected'] = 2.0 + (amount / 50) + (predict_gpa(current_user)['efficiency'] / 200)
            result['current'] = predict_gpa(current_user)['gpa']
        elif scenario == 'exercise':
            result['projected'] = amount * 30 * 30
            result['current'] = sum(f.duration_min for f in current_user.fitness_logs)

    return render_template('user/simulation.html',
                           user=current_user,
                           result=result,
                           current_values=current_values)


@app.route('/user/assistant', methods=['GET', 'POST'])
@login_required
def user_assistant():
    # Load history item
    load_id = request.args.get('load')
    if load_id:
        msg = ChatMessage.query.filter_by(id=load_id, user_id=current_user.id).first()
        if msg:
            session['last_query'] = msg.question
            session['last_response'] = msg.response
        else:
            flash('Message not found.')
        return redirect(url_for('user_assistant'))

    # Handle new message
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if not query:
            flash('Please enter a question.')
            return redirect(url_for('user_assistant'))

        response = None

        # ---- Try Gemini ----
        if gemini_model and GEMINI_API_KEY:
            try:
                user_data = f"User: {current_user.username}, age {current_user.age}, savings ₹{current_user.savings}"
                prompt = f"{user_data}\nQuestion: {query}\nAnswer concisely (max 150 words)."
                gemini_response = gemini_model.generate_content(prompt)
                response = gemini_response.text.strip()
                print("Gemini response OK")
            except Exception as e:
                print(f"Gemini error: {e}")
                flash('AI service unavailable. Using fallback.')
                response = None

        # ---- Fallback ----
        if not response:
            q = query.lower()
            if 'spend' in q or 'save' in q or 'money' in q:
                metrics = calculate_financial_metrics(current_user)
                response = f"Your savings rate is {metrics['rate']:.1f}%. I recommend saving at least 30% of income."
            elif 'study' in q or 'gpa' in q:
                gpa = predict_gpa(current_user)
                response = f"Your predicted GPA is {gpa['gpa']}/4.0. {gpa['recommendation']}"
            elif 'fitness' in q or 'exercise' in q:
                total = sum(f.duration_min for f in current_user.fitness_logs)
                response = f"You've logged {total} minutes of exercise. Aim for 150 min/week."
            elif 'goal' in q:
                response = "I can help you set SMART goals. What area would you like to focus on?"
            else:
                response = "I'm your Digital Twin AI. Ask about finances, studies, fitness, or goals!"

        # ---- Final safety ----
        if not response:
            response = "I'm sorry, I couldn't generate a response. Please try again."

        # ---- Save ----
        chat_msg = ChatMessage(user_id=current_user.id, question=query, response=response)
        db.session.add(chat_msg)
        db.session.commit()

        session['last_query'] = query
        session['last_response'] = response

        print(f"Saved response: {response[:50]}...")  # debug
        return redirect(url_for('user_assistant'))

    # ---- GET: show chat ----
    last_query = session.get('last_query')
    last_response = session.get('last_response')
    all_messages = ChatMessage.query.filter_by(user_id=current_user.id).order_by(ChatMessage.timestamp.desc()).all()

    return render_template('user/assistant.html',
                           user=current_user,
                           last_query=last_query,
                           last_response=last_response,
                           all_messages=all_messages)

@app.route('/user/add_goal', methods=['POST'])
@login_required
def add_goal():

    description = request.form.get('description', '').strip()
    target_date_str = request.form.get('target_date', '')

    if not description:
        flash('Goal description is required.')
        return redirect(url_for('user_dashboard'))

    target_date = None
    if target_date_str:
        try:
            target_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()
        except ValueError:
            flash('Invalid date format.')
            return redirect(url_for('user_dashboard'))

    goal = Goal(
        user_id=current_user.id,
        description=description,
        target_date=target_date,
        achieved=False
    )
    db.session.add(goal)
    db.session.commit()

    flash('Goal added successfully! 🎯')
    return redirect(url_for('user_dashboard'))
# ------------------------- Admin Routes (unchanged) -------------------------
@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash('Admin access required.')
        return redirect(url_for('user_dashboard'))

    # ----- Basic counts -----
    total_users = User.query.count()
    total_transactions = Transaction.query.count()
    total_study = StudyLog.query.count()
    total_fitness = FitnessLog.query.count()

    # ----- User growth (monthly registrations for last 6 months) -----
    from collections import defaultdict
    monthly_users = defaultdict(int)
    for user in User.query.all():
        month_key = user.created_at.strftime('%Y-%m')
        monthly_users[month_key] += 1
    months = sorted(monthly_users.keys())[-6:]
    user_counts = [monthly_users.get(m, 0) for m in months]

    # ----- Transaction summary (income vs expense totals) -----
    total_income = db.session.query(db.func.sum(Transaction.amount)).filter(Transaction.type == 'income').scalar() or 0
    total_expense = db.session.query(db.func.sum(Transaction.amount)).filter(Transaction.type == 'expense').scalar() or 0

    # ----- Most active users (based on transaction count) -----
    from sqlalchemy import func
    active_users = db.session.query(
        User.username,
        func.count(Transaction.id).label('tx_count')
    ).join(Transaction).group_by(User.id).order_by(func.count(Transaction.id).desc()).limit(5).all()

    # ----- Top study subjects overall -----
    subject_counts = db.session.query(
        StudyLog.subject,
        func.sum(StudyLog.hours).label('total_hours')
    ).filter(StudyLog.subject != '').group_by(StudyLog.subject).order_by(func.sum(StudyLog.hours).desc()).limit(5).all()

    # ----- Top fitness activities overall -----
    activity_counts = db.session.query(
        FitnessLog.activity,
        func.sum(FitnessLog.duration_min).label('total_min')
    ).group_by(FitnessLog.activity).order_by(func.sum(FitnessLog.duration_min).desc()).limit(5).all()

    # ----- Average savings per user (from User model) -----
    avg_savings = db.session.query(func.avg(User.savings)).scalar() or 0

    # ----- Recent activity across all users (last 5 entries) -----
    recent_activity = []
    # Combine transactions, study, fitness logs in one list with type
    transactions = Transaction.query.order_by(Transaction.date.desc()).limit(5).all()
    for tx in transactions:
        recent_activity.append({
            'type': 'transaction',
            'user': tx.user.username,
            'date': tx.date,
            'detail': f"{tx.type}: {tx.category} ₹{tx.amount}"
        })
    study_logs = StudyLog.query.order_by(StudyLog.date.desc()).limit(5).all()
    for log in study_logs:
        recent_activity.append({
            'type': 'study',
            'user': log.user.username,
            'date': log.date,
            'detail': f"{log.hours}h - {log.subject} ({log.productivity_score}%)"
        })
    fitness_logs = FitnessLog.query.order_by(FitnessLog.date.desc()).limit(5).all()
    for log in fitness_logs:
        recent_activity.append({
            'type': 'fitness',
            'user': log.user.username,
            'date': log.date,
            'detail': f"{log.activity} {log.duration_min}min"
        })
    # Sort all by date descending and take top 10
    recent_activity = sorted(recent_activity, key=lambda x: x['date'], reverse=True)[:10]

    # ----- Additional stats -----
    total_goals = Goal.query.count()
    achieved_goals = Goal.query.filter_by(achieved=True).count()
    goal_completion_rate = (achieved_goals / total_goals * 100) if total_goals > 0 else 0

    return render_template('admin/dashboard.html',
                           total_users=total_users,
                           total_transactions=total_transactions,
                           total_study=total_study,
                           total_fitness=total_fitness,
                           months=months,
                           user_counts=user_counts,
                           total_income=total_income,
                           total_expense=total_expense,
                           active_users=active_users,
                           subject_counts=subject_counts,
                           activity_counts=activity_counts,
                           avg_savings=avg_savings,
                           recent_activity=recent_activity,
                           total_goals=total_goals,
                           goal_completion_rate=goal_completion_rate)


@app.route('/admin/users')
@login_required
def admin_users():
    if not current_user.is_admin:
        flash('Admin access required.')
        return redirect(url_for('user_dashboard'))
    users = User.query.all()
    return render_template('admin/users.html', users=users)


@app.route('/admin/user/<int:user_id>')
@login_required
def admin_user_detail(user_id):
    if not current_user.is_admin:
        flash('Admin access required.')
        return redirect(url_for('user_dashboard'))

    user = User.query.get_or_404(user_id)
    metrics = calculate_financial_metrics(user)
    gpa = predict_gpa(user)
    total_fitness = sum(f.duration_min for f in user.fitness_logs)

    # ---- Additional stats ----
    total_transactions = len(user.transactions)
    total_study_logs = len(user.study_logs)
    total_fitness_logs = len(user.fitness_logs)
    total_goals = len(user.goals)
    achieved_goals = len([g for g in user.goals if g.achieved])

    # ---- Charts data ----
    from collections import defaultdict

    # Monthly income/expenses
    monthly_income = defaultdict(float)
    monthly_expenses = defaultdict(float)
    for tx in user.transactions:
        month_key = tx.date.strftime('%Y-%m')
        if tx.type == 'income':
            monthly_income[month_key] += tx.amount
        else:
            monthly_expenses[month_key] += tx.amount
    months = sorted(set(monthly_income.keys()) | set(monthly_expenses.keys()))
    months = months[-6:]  # last 6 months
    income_data = [monthly_income.get(m, 0) for m in months]
    expense_data = [monthly_expenses.get(m, 0) for m in months]

    # Study hours last 14 days
    study_dates = defaultdict(float)
    for log in user.study_logs:
        study_dates[log.date.strftime('%Y-%m-%d')] += log.hours
    study_days = sorted(study_dates.keys())[-14:]
    study_hours = [study_dates.get(d, 0) for d in study_days]

    # Fitness minutes last 14 days
    fitness_dates = defaultdict(float)
    for log in user.fitness_logs:
        fitness_dates[log.date.strftime('%Y-%m-%d')] += log.duration_min
    fitness_days = sorted(fitness_dates.keys())[-14:]
    fitness_minutes = [fitness_dates.get(d, 0) for d in fitness_days]

    # Recent activity (last 5 each)
    recent_transactions = user.transactions[-5:][::-1]  # newest first
    recent_study = user.study_logs[-5:][::-1]
    recent_fitness = user.fitness_logs[-5:][::-1]

    return render_template('admin/user_detail.html',
                           user=user,
                           metrics=metrics,
                           gpa=gpa,
                           total_fitness=total_fitness,
                           total_transactions=total_transactions,
                           total_study_logs=total_study_logs,
                           total_fitness_logs=total_fitness_logs,
                           total_goals=total_goals,
                           achieved_goals=achieved_goals,
                           months=months,
                           income_data=income_data,
                           expense_data=expense_data,
                           study_days=study_days,
                           study_hours=study_hours,
                           fitness_days=fitness_days,
                           fitness_minutes=fitness_minutes,
                           recent_transactions=recent_transactions,
                           recent_study=recent_study,
                           recent_fitness=recent_fitness)
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        subject = request.form.get('subject', '').strip()
        message = request.form.get('message', '').strip()

        if not all([name, email, subject, message]):
            flash('All fields are required.')
            return redirect(url_for('contact'))

        # Here you can send an email, save to database, etc.
        # For now, we'll just flash a success message.
        flash('✅ Your message has been sent successfully! We\'ll get back to you soon.')
        return redirect(url_for('contact'))

    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/privacy')
def privacy():
    return render_template('privacy.html')
@app.route('/terms')
def terms():
    return render_template('terms.html')


@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html')

# ------------------------- Run the App -------------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        create_default_admin()   # <-- ADD THIS LINE
    app.run(debug=True)