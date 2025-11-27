# backend/job_descriptions.py
"""
Predefined job descriptions for different roles
"""

JOB_DESCRIPTIONS = {
    "software_engineer": """
Software Engineer Position - Python, Django, React, SQL, Git, Docker

Required Skills:
- Python programming (2+ years experience)
- Django or Flask web frameworks
- React.js or similar frontend framework
- SQL database design and queries
- Git version control
- REST API development
- Object-oriented programming (OOP)
- Data structures and algorithms
- MongoDB or PostgreSQL
- Docker containerization
- Agile development methodologies
- Code review and testing

Technical Requirements:
- Experience with web development frameworks
- Database design and optimization
- API development and integration
- Version control with Git
- Cloud platforms (AWS, Azure, or GCP)
- CI/CD pipeline experience
- Unit testing and debugging
- Software architecture principles

Responsibilities:
- Develop and maintain web applications using Python and Django
- Build responsive frontend interfaces with React
- Design and optimize database schemas
- Implement RESTful APIs
- Write clean, maintainable, and testable code
- Collaborate with cross-functional teams
- Participate in code reviews and technical discussions
""",

    "data_scientist": """
Data Scientist Position - Python, Machine Learning, Statistics, Pandas, Scikit-learn

Required Skills:
- Python programming for data analysis
- Pandas, NumPy for data manipulation
- Scikit-learn, TensorFlow, or PyTorch for machine learning
- Statistics and probability theory
- Data visualization with Matplotlib, Seaborn
- SQL for database queries
- Jupyter Notebooks
- Git version control
- Machine learning algorithms
- Data preprocessing and cleaning
- Statistical modeling
- A/B testing and experimentation

Technical Requirements:
- Experience with machine learning libraries
- Data visualization and reporting
- Database querying and data extraction
- Statistical analysis and hypothesis testing
- Model evaluation and validation
- Feature engineering and selection
- Data pipeline development
- Cloud platforms (AWS, Azure, GCP)

Responsibilities:
- Analyze large datasets using Python and statistical methods
- Build and deploy machine learning models
- Create data visualizations and dashboards
- Perform statistical analysis and A/B testing
- Clean and preprocess data for analysis
- Collaborate with engineering teams on data pipelines
- Present findings to stakeholders through reports and presentations
""",

    "product_manager": """
We are looking for an Associate Product Manager (1–2 years of experience).
The ideal candidate should have:
- Basic understanding of the product lifecycle and agile methodologies.
- Strong communication and organizational skills.
- Experience in documenting requirements and coordinating with technical teams.
- Familiarity with UI/UX concepts and customer experience design.
- Internship or junior-level product management experience (up to 2 years).

Responsibilities:
- Assist senior product managers in defining product strategy and roadmap.
- Collaborate with developers, designers, and QA teams to deliver features.
- Collect user feedback and prepare reports to improve product usability.
""",

    "devops_engineer": """
We are hiring a Junior DevOps Engineer (1–2 years of experience).
The ideal candidate should have:
- Hands-on knowledge of Linux/Unix systems and shell scripting.
- Familiarity with cloud platforms (AWS, Azure, or GCP).
- Experience with version control systems (Git) and CI/CD pipelines.
- Exposure to containerization (Docker, Kubernetes is a plus).
- Internship or 1–2 years of experience in cloud/DevOps environments.

Responsibilities:
- Maintain CI/CD pipelines and assist in infrastructure automation.
- Monitor and troubleshoot deployments in cloud environments.
- Collaborate with developers to optimize build and release processes.
""",

    "ui_ux_designer": """
We are looking for a Junior UI/UX Designer (1–2 years of experience).
The ideal candidate should have:
- Strong creativity and attention to detail.
- Proficiency in design tools (Figma, Adobe XD, Sketch, Photoshop).
- Ability to translate business requirements into wireframes and prototypes.
- Understanding of user-centered design principles and accessibility standards.
- Prior internship or 1–2 years of experience in UI/UX design.

Responsibilities:
- Create wireframes, prototypes, and high-fidelity designs.
- Conduct user research and usability testing.
- Work with developers to ensure design consistency and user satisfaction.
"""
}

def get_job_description(job_role: str) -> str:
    """Get job description for a specific role"""
    return JOB_DESCRIPTIONS.get(job_role.lower(), JOB_DESCRIPTIONS["software_engineer"])

def get_available_roles() -> list:
    """Get list of available job roles"""
    return list(JOB_DESCRIPTIONS.keys())
