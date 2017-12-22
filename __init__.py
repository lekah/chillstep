import traceback
import numpy as np
from aiida.orm import JobCalculation, Calculation, Data, Code
from aiida.common.exceptions import InputValidationError, ModificationNotAllowed
from abc import abstractmethod
from aiida.backends.utils import get_automatic_user
from aiida.orm.querybuilder import QueryBuilder
from aiida.common.datastructures import calc_states, sort_states
from aiida.backends import settings
from aiida.backends.utils import BACKEND_DJANGO, BACKEND_SQLA
from aiida.common.links import LinkType
from aiida.common.hashing import make_hash

PAUSE_ATTRIBUTE_KEY = '_PAUSED'
PAUSED_STATE = 'PAUSED'

def _set_state_dj(self, state):
    """
    Set the state of the calculation.

    Set it in the DbCalcState to have also the uniqueness check.
    Moreover (except for the IMPORTED state) also store in the 'state'
    attribute, useful to know it also after importing, and for faster
    querying.

    .. todo:: Add further checks to enforce that the states are set
       in order?

    :param state: a string with the state. This must be a valid string,
      from ``aiida.common.datastructures.calc_states``.
    :raise: ModificationNotAllowed if the given state was already set.
    """
    from django.db import transaction, IntegrityError

    from aiida.common.datastructures import sort_states
    from aiida.backends.djsite.db.models import DbCalcState

    if not self.is_stored:
        raise ModificationNotAllowed("Cannot set the calculation state "
                                     "before storing")

    if state not in calc_states:
        raise ValueError(
            "'{}' is not a valid calculation status".format(state))

    old_state = self.get_state()
    if old_state:
        state_sequence = [state, old_state]

        # sort from new to old: if they are equal, then it is a valid
        # advance in state (otherwise, we are going backwards...)
        if sort_states(state_sequence) != state_sequence:
            raise ModificationNotAllowed("Cannot change the state from {} "
                                         "to {}".format(old_state, state))

    try:
        with transaction.atomic():
            new_state = DbCalcState(dbnode=self.dbnode, state=state).save()
    except IntegrityError:
        raise ModificationNotAllowed(
            "Calculation pk= {} already transited through "
            "the state {}".format(self.pk, state))

    # For non-imported states, also set in the attribute (so that, if we
    # export, we can still see the original state the calculation had.
    if state != calc_states.IMPORTED:
        self._set_attr('state', state)

def get_state_dj(self, from_attribute=False):
    """
    Get the state of the calculation.

    .. note:: this method returns the NOTFOUND state if no state
      is found in the DB.

    .. note:: the 'most recent' state is obtained using the logic in the
      ``aiida.common.datastructures.sort_states`` function.

    .. todo:: Understand if the state returned when no state entry is found
      in the DB is the best choice.

    :param from_attribute: if set to True, read it from the attributes
      (the attribute is also set with set_state, unless the state is set
      to IMPORTED; in this way we can also see the state before storing).

    :return: a string. If from_attribute is True and no attribute is found,
      return None. If from_attribute is False and no entry is found in the
      DB, return the "NOTFOUND" state.
    """
    from aiida.backends.djsite.db.models import DbCalcState

    if from_attribute:
        return self.get_attr('state', None)
    else:
        if not self.is_stored:
            return calc_states.NEW
        else:
            this_calc_states = DbCalcState.objects.filter(
                dbnode=self).values_list('state', flat=True)
            if not this_calc_states:
                return None
            else:
                try:
                    most_recent_state = sort_states(this_calc_states)[0]
                except ValueError as e:
                    raise DbContentError("Error in the content of the "
                                         "DbCalcState table ({})".format(
                        e.message))

                return most_recent_state



class Context(object):
    def __init__(self, node):
        super(Context, self).__setattr__('_node', node)

    def __setattr__(self, name, value):
        self._node._set_attr(name, value)
    def __getattr__(self, name):
        return self._node.get_attr(name)
    def __delattr__(self, name):
        self._node._del_attr(name)

class ChillstepCalculation(Calculation):

    def __init__(self, dbnode=None, **kwargs):
        # Everything that is input has to be Data
        super(ChillstepCalculation, self).__init__(dbnode=dbnode)
        for k, v in kwargs.items():
            if not isinstance(v, (Code, Data)):
                raise InputValidationError("Input to a Chillstep has to be Data or Code")
            # These are the inputs:
            self.add_link_from(v, label=k, link_type=LinkType.INPUT)
        self.ctx = Context(self)
        if dbnode is None:
            #This was just created:
            self.goto(self.start)


    def get_inputs_dict(self):
        return {k:v for k, v in super(ChillstepCalculation, self).get_inputs_dict().items() if  isinstance(v, (Data, Code))}
    @property
    def inputs(self):
        return self.inp

    def goto(self, func):
        #~ self._set_attr('_next', func.__name__)
        try:
            self.ctx._last = self.ctx._next
        except:
            self.ctx._last = None
        self.ctx._next = func.__name__
    @abstractmethod
    def start(self):
        print "starting"
    def exit(self):
        pass

    def submit(self):
        if self._to_be_stored:
            self.store_all()
        self._set_state(calc_states.WITHSCHEDULER)


    def _validate(self):
        ###
        super(Calculation, self)._validate()

    def _set_state(self, state):
        if settings.BACKEND == BACKEND_DJANGO:
            _set_state_dj(self, state)
        elif settings.BACKEND == BACKEND_SQLA:
            _set_state_sqla(self, state)
        else:
            raise Exception("unknown backend {}".format(settings.BACKEND))
    def get_state(self):
        if self.get_attr(PAUSE_ATTRIBUTE_KEY, False):
            return PAUSED_STATE
        if settings.BACKEND == BACKEND_DJANGO:
            return get_state_dj(self)
        elif settings.BACKEND == BACKEND_SQLA:
            return get_state_sqla(self)
        else:
            raise Exception("unknown backend {}".format(settings.BACKEND))


    def get_scheduler_state(self):
        return "RUNNING"

    def get_scheduler_output(self):
        return None

    def get_scheduler_error(self):
        return None

    def _linking_as_output(self, dest, link_type):
        """
        :note: Further checks, such as that the output data type is 'Data',
          are done in the super() class.

        :param dest: a Data object instance of the database
        :raise: ValueError if a link from self to dest is not allowed.
        """
        valid_states = [
            calc_states.NEW,
            calc_states.WITHSCHEDULER,
        ]

        if self.get_state() not in valid_states:
            raise ModificationNotAllowed(
                "Can add an output node to a calculation only if it is in one "
                "of the following states: {}, it is instead {}".format(
                    valid_states, self.get_state()))

        return super(Calculation, self)._linking_as_output(dest, link_type)

    def pause(self):
        to_pause = self.get_running_slaves() + [self]
        for cs in to_pause:
            prev_state = cs.get_state()
            if prev_state == PAUSED_STATE:
                print "  ", cs, "was already paused"
            else:
                print "   Pausing", cs, "(previous state was {})".format(prev_state)
                cs._set_attr(PAUSE_ATTRIBUTE_KEY, True)

    def get_running_slaves(self, projections=['*']):
        return [_ for _,  in QueryBuilder().append(
                ChillstepCalculation, filters={'id':self.id},tag='parent').append(
                Calculation, filters={'state':{'!in':['FINISHED', 'FAILED', 'SUBMISSIONFAILED', 'PARSINGFAILED']}}, output_of='parent', project=projections,
                edge_filters={'type':LinkType.CALL.value}
            ).all()]


def run(cs, store=False):
    while True:
        finished = tick_chillstepper(cs, store=store)
        if finished:
            break


def tick_chillstepper(cs, dry_run=False, backoff=True):

    try:
        last_funcname = cs.get_attr('_last', None)
        funcname = cs.get_attr('_next')
        print '@{} {} [ {} ] ( {} )'.format(cs.__class__.__name__, cs.uuid, cs.pk, last_funcname)
        if cs.get_attr(PAUSE_ATTRIBUTE_KEY, False):
            print "    is paused"
            return
        try:
            backoff_counter = cs.get_attr('backoff_counter', 0)
        except AttributeError:
            backoff_counter = 0
        if backoff and backoff_counter > 0:
            print "    Backoff counter is {}".format(backoff_counter)
            if np.random.random() < 0.5**backoff_counter:
                print "    I will try to rerun"
            else:
                print "    Has to wait due to backoff"
                return

        waiting_for_pks = cs.get_running_slaves(projections=['id'])
        if len(waiting_for_pks):
            print "   Waiting for:", ' '.join(map(str, waiting_for_pks))
        else:
            # What's next to do?
            if funcname == 'exit':
                print "   FINISHED"
                cs._set_state(calc_states.FINISHED)
            else:
                print "   Reaching next step ( {} )".format(funcname)
                returned = getattr(cs, funcname)()
                if returned is None:
                    return False
                for k, v in returned.items():
                    if not dry_run:
                        if isinstance(v, Data):
                            print "      Adding {} as returned ( {} )".format(v, k)
                            assert v.is_stored, "Received unstored Data instance"
                            v.add_link_from(cs,  label=k, link_type=LinkType.RETURN)
                        elif isinstance(v, Calculation):
                            v.add_link_from(cs,  label=k, link_type=LinkType.CALL)
                            if isinstance(v, (JobCalculation, ChillstepCalculation)):
                                v.store_all()
                                v.submit()
                            print "      Adding {} as called ( {} )".format(v, k)
                        else:
                            raise Exception("Unspecified type {}".format(type(v)))

    except Exception as e:
        trcback = str(traceback.format_exc())
        msg = "ERROR ! This Chillstepper got an error for {} in the {} method, we report down the stack trace:\n{}".format(
                cs, funcname, trcback)
        print trcback
        if dry_run:
            raise
        cs._set_state(calc_states.FAILED)
        cs._set_attr('fail_hash', make_hash(trcback))
        cs._set_attr('fail_msg', msg)
        cs.add_comment(trcback, user=get_automatic_user())



def tick_all(dry_run=False, pk=None, backoff=True):
    qb = QueryBuilder()
    qb.append(ChillstepCalculation,tag='c', filters={'state':calc_states.WITHSCHEDULER})
    if isinstance(pk, int):
        qb.add_filter('c', {'id':pk})
    elif isinstance(pk, (list, tuple)):
        qb.add_filter('c', {'id':{'in':pk}})
    for chillstepcalc, in qb.all():
        tick_chillstepper(chillstepcalc, dry_run=dry_run, backoff=backoff)



if __name__ == '__main__':
    tick_all()
