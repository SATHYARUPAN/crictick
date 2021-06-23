#Database testing ---> python -m pytest
import managed_db

def database():
	managed_db.create_usertable()
	sample_data = [
	('sathya','rupan'),
	('sheik','sultan')
	]

	managed_db.add_userdata(sample_data)

def test_database():
	assert len(list(managed_db.view_all_users())) == 2


