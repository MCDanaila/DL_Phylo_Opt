import re
import random
import os
from subprocess import Popen, PIPE, STDOUT

SEP = '/'
RAXML_NG_SCRIPT = "/Users/mihaid/Coding-Projects/thesis/tools/raxml-ng_v1.2.0_macos_x86_64/raxml-ng"

def parse_raxmlNG_content(content):
	"""
	:return: dictionary with the attributes - string typed. if parameter was not estimated, empty string
	"""
	res_dict = dict.fromkeys(["ll", "pInv", "gamma",
							  "fA", "fC", "fG", "fT",
							  "subAC", "subAG", "subAT", "subCG", "subCT", "subGT",
							  "time"], "")

	# likelihood
	ll_re = re.search("Final LogLikelihood:\s+(.*)", content)
	print("ll_re: ", ll_re)
	if ll_re:
		res_dict["ll"] = ll_re.group(1).strip()
	elif re.search("BL opt converged to a worse likelihood score by", content) or re.search("failed", content):   # temp, till next version is available
		ll_ini = re.search("initial LogLikelihood:\s+(.*)", content)
		if ll_ini:
			res_dict["ll"] = ll_ini.group(1).strip()
	else:
		res_dict["ll"] = 'unknown raxml-ng error, check "parse_raxmlNG_content" function'


	# gamma (alpha parameter) and proportion of invariant sites
	gamma_regex = re.search("alpha:\s+(\d+\.?\d*)\s+", content)
	pinv_regex = re.search("P-inv.*:\s+(\d+\.?\d*)", content)
	if gamma_regex:
		res_dict['gamma'] = gamma_regex.group(1).strip()
	if pinv_regex:
		res_dict['pInv'] = pinv_regex.group(1).strip()

	# Nucleotides frequencies
	nucs_freq = re.search("Base frequencies.*?:\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)", content)
	if nucs_freq:
		for i,nuc in enumerate("ACGT"):
			res_dict["f" + nuc] = nucs_freq.group(i+1).strip()

	# substitution frequencies
	subs_freq = re.search("Substitution rates.*:\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)", content)
	if subs_freq:
		for i,nuc_pair in enumerate(["AC", "AG", "AT", "CG", "CT", "GT"]):  # todo: make sure order
			res_dict["sub" + nuc_pair] = subs_freq.group(i+1).strip()

	# Elapsed time of raxml-ng optimization
	rtime = re.search("Elapsed time:\s+(\d+\.?\d*)\s+seconds", content)
	if rtime:
		res_dict["time"] = rtime.group(1).strip()
	else:
		res_dict["time"] = 'no ll opt_no time'

	return res_dict


def call_raxml_mem(tree_str, msa_tmpfile, rates, pinv, alpha, freq):
	model_line_params = 'GTR{rates}+I{pinv}+G{alpha}+F{freq}'.format(rates="{{{0}}}".format("/".join(rates)),
									 pinv="{{{0}}}".format(pinv), alpha="{{{0}}}".format(alpha),
									 freq="{{{0}}}".format("/".join(freq)))

	# create tree file in memory and not in the storage:
	tree_rampath = "/Users/mihaid/" + str(random.random())  + str(random.random()) + "tree"  # the var is the str: tmp{dir_suffix}

	try:
		with open(tree_rampath, "w") as fpw:
			fpw.write(tree_str)

		p = Popen([RAXML_NG_SCRIPT, '--evaluate', '--msa', msa_tmpfile,'--threads', '2', '--opt-branches', 'on', '--opt-model', 'off', '--model', model_line_params, '--nofiles', '--tree', tree_rampath], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
		raxml_stdout = p.communicate()[0]
		raxml_output = raxml_stdout.decode()

		res_dict = parse_raxmlNG_content(raxml_output)
		ll = res_dict['ll']
		rtime = res_dict['time']

	except Exception as e:
		print(msa_tmpfile.split(SEP)[-1][3:])
		print(e)
		exit()
	finally:
		os.remove(tree_rampath)

	return ll, rtime