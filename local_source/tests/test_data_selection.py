import unittest
from copy import deepcopy
from src.data.common import digest
from src.data.selection import select_manifest


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.cfg={'key':'fixture'}
        self.manifest={'config_hash':digest(self.cfg),'episodes':[{'episode':'a'},{'episode':'b'}],'rejected':[]}
        self.manifest['manifest_hash']=digest(self.manifest)
        self.selection={'source_manifest_hash':self.manifest['manifest_hash'],'episodes':['a']}

    def test_subset_provenance_and_no_mutation(self):
        before=deepcopy(self.manifest)
        result=select_manifest(self.manifest,self.selection,self.cfg)
        self.assertEqual(self.manifest,before)
        self.assertEqual(result['episodes'],[{'episode':'a'}])
        self.assertEqual(result['selection']['excluded_accepted_episodes'],['b'])
        self.assertEqual(result['manifest_hash'],digest({k:v for k,v in result.items() if k!='manifest_hash'}))

    def test_unknown_duplicate_empty_and_stale_rejected(self):
        for ids in [['c'],['a','a'],[]]:
            selection={**self.selection,'episodes':ids}
            with self.assertRaises(ValueError): select_manifest(self.manifest,selection,self.cfg)
        stale={**self.selection,'source_manifest_hash':'old'}
        with self.assertRaises(ValueError): select_manifest(self.manifest,stale,self.cfg)
        tampered={**self.manifest,'rejected':[{'episode':'x'}]}
        with self.assertRaises(ValueError): select_manifest(tampered,self.selection,self.cfg)


if __name__=='__main__': unittest.main()
