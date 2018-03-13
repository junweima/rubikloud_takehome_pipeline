""" Rubikloud take home problem """
import luigi
import pandas as pd
import sklearn as sk
import pdb

class CleanDataTask(luigi.Task):
	""" Cleans the input CSV file by removing any rows without valid geo-coordinates.

		Output file should contain just the rows that have geo-coordinates and
		non-(0.0, 0.0) files.
	"""
	print('task1')
	tweet_file = luigi.Parameter()
	output_file = luigi.Parameter(default='clean_data.csv')

	def requires(self):
		return []

	def output(self):
		return luigi.LocalTarget(self.output_file)

	def run(self):
		df = pd.read_csv(self.tweet_file, encoding='iso8859_5')
		with self.output().open('w') as f:
			df_clean = df[['airline_sentiment', 'tweet_coord']][~df['tweet_coord'].isin(['[0.0, 0.0]'])]
			df_clean = df_clean[df['tweet_coord'].notnull()]
			f.write(df_clean.to_string())
			

class TrainingDataTask(luigi.Task):
	""" Extracts features/outcome variable in preparation for training a model.

		Output file should have columns corresponding to the training data:
		- y = airline_sentiment (coded as 0=negative, 1=neutral, 2=positive)
		- X = a one-hot coded column for each city in "cities.csv"
	"""
	print('task2')
	tweet_file = luigi.Parameter()
	cities_file = luigi.Parameter(default='cities.csv')
	output_file = luigi.Parameter(default='features.csv')

	def requires(self):
		return CleanDataTask(self.tweet_file)

	def output(self):
		return luigi.LocalTarget(self.output_file)

	def run(self):
		with self.output().open('w') as f:
			f.write('something for testing part 2')
		


class TrainModelTask(luigi.Task):
	""" Trains a classifier to predict negative, neutral, positive
		based only on the input city.

		Output file should be the pickle'd model.
	"""
	print('task3')
	tweet_file = luigi.Parameter()
	output_file = luigi.Parameter(default='model.pkl')

	def requires(self):
		return TrainingDataTask(self.tweet_file)

	def output(self):
		return luigi.LocalTarget(self.output_file)

	def run(self):
		with self.output().open('w') as f:
			f.write('something for testing part 3')


class ScoreTask(luigi.Task):
	""" Uses the scored model to compute the sentiment for each city.

		Output file should be a four column CSV with columns:
		- city name
		- negative probability
		- neutral probability
		- positive probability
	"""
	print('task4')
	tweet_file = luigi.Parameter()
	output_file = luigi.Parameter(default='scores.csv')

	def requires(self):
		return TrainModelTask(self.tweet_file)

	def output(self):
		return luigi.LocalTarget(self.output_file)

	def run(self):
		with self.output().open('w') as f:
			f.write('something for testing part 4')


if __name__ == "__main__":
	luigi.run()
