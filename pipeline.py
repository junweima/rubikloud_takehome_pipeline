""" Rubikloud take home problem """
import luigi
import pandas as pd
import sklearn as sk
import os
import re
import numpy as np
import pdb

class CleanDataTask(luigi.Task):
	""" Cleans the input CSV file by removing any rows without valid geo-coordinates.

		Output file should contain just the rows that have geo-coordinates and
		non-(0.0, 0.0) files.
	"""
	tweet_file = luigi.Parameter()
	output_file = luigi.Parameter(default='clean_data.csv')

	def requires(self):
		return []

	def output(self):
		return luigi.LocalTarget(self.output_file)

	def run(self):

		df = pd.read_csv(self.tweet_file, encoding='iso8859_5')
		df_clean = df[df['tweet_coord'].notnull()]
		df_clean = df_clean[['airline_sentiment', 'tweet_coord']][~df_clean['tweet_coord'].isin(['[0.0, 0.0]'])]
		df_clean['latitude'] = df_clean.apply(lambda row: re.findall(r'[0-9\.]+', row['tweet_coord'])[0], axis=1)
		df_clean['longitude'] = df_clean.apply(lambda row: re.findall(r'[0-9\.]+', row['tweet_coord'])[1], axis=1)

		if not os.path.isfile(self.output_file):
			f = open(self.output_file, 'w')
			f.close()
		df_clean.to_csv(self.output_file, sep=',')
			

class TrainingDataTask(luigi.Task):
	""" Extracts features/outcome variable in preparation for training a model.

		Output file should have columns corresponding to the training data:
		- y = airline_sentiment (coded as 0=negative, 1=neutral, 2=positive)
		- X = a one-hot coded column for each city in "cities.csv"
	"""
	tweet_file = luigi.Parameter()
	cities_file = luigi.Parameter(default='cities.csv')
	output_file = luigi.Parameter(default='features.csv')

	def requires(self):
		return CleanDataTask(self.tweet_file)

	def output(self):
		return luigi.LocalTarget(self.output_file)

	def run(self):
		
		one_hot = lambda x, k: np.array(x == np.arange(k)[None, :], dtype=int)

		df = pd.read_csv(self.input().open().name)
		df.loc[df.airline_sentiment == 'negative', 'airline_sentiment'] = 0
		df.loc[df.airline_sentiment == 'neutral', 'airline_sentiment'] = 1
		df.loc[df.airline_sentiment == 'positive', 'airline_sentiment'] = 2

		df_cities = pd.read_csv(self.cities_file)
		nearest_cities = []
		for i in range(df.shape[0]):
			nearest_city = df_cities.index[((df_cities['latitude']-df['latitude'][i]).pow(2) + (df_cities['longitude']-df['longitude'][i]).pow(2)).argsort()[0]]
			nearest_cities.append(one_hot(nearest_city, df_cities.shape[0]).tolist())
			# nearest_cities.append(nearest_city)
		
		df_nearest_cities = pd.DataFrame({'nearest_city': nearest_cities})

		df = pd.concat([df['airline_sentiment'], df_nearest_cities['nearest_city']], axis=1)
		df.columns = ['y', 'x']

		if not os.path.isfile(self.output_file):
			f = open(self.output_file, 'w')
			f.close()
		df.to_csv(self.output_file, sep=',')

		# df_test = pd.read_csv(self.output_file)
		# pdb.set_trace()


class TrainModelTask(luigi.Task):
	""" Trains a classifier to predict negative, neutral, positive
		based only on the input city.

		Output file should be the pickle'd model.
	"""
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
